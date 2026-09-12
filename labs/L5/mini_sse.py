#!/usr/bin/env python3
"""SSE 编解码最小实现：只处理 body 字节，不碰 HTTP chunk 分帧。

规范里容易漏的四条：
  1. 行分隔符 CR、LF、CRLF 都算；CRLF 可能被拆到两次 recv 里。
  2. 冒号后紧跟的**一个**空格被吃掉，多出来的属于 data。
  3. 以冒号开头的行是注释，客户端必须忽略——vLLM 的 keep-alive 就走这条路。
  4. EOF 不是事件结束符：没等到空行就结束的半个事件要丢掉。
"""
import codecs
import json
import random
import re

_LINES = re.compile(r"\r\n|\r|\n")


class SSEDecoder:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self.buffer = ""
        self.data = []
        self.kind = "message"
        self.last_id = ""
        self.retry = None

    def line(self, line):
        if line == "":
            event = None
            if self.data:
                event = {"event": self.kind, "data": "\n".join(self.data), "id": self.last_id}
            self.data, self.kind = [], "message"
            return event
        if line.startswith(":"):
            return None
        key, colon, value = line.partition(":")
        if not colon:
            value = ""
        if value.startswith(" "):
            value = value[1:]
        if key == "data":
            self.data.append(value)
        elif key == "event":
            self.kind = value
        elif key == "id" and "\x00" not in value:
            self.last_id = value
        elif key == "retry" and value.isascii() and value.isdecimal():
            self.retry = int(value)
        return None

    def feed(self, body_bytes, final=False):
        self.buffer += self.decoder.decode(body_bytes, final=final)
        events = []
        while True:
            positions = [p for p in (self.buffer.find("\r"), self.buffer.find("\n")) if p >= 0]
            if not positions:
                break
            end = min(positions)
            # CR might be the first half of a CRLF split over two reads.
            if self.buffer[end] == "\r" and end + 1 == len(self.buffer) and not final:
                break
            width = 2 if self.buffer[end:end + 2] == "\r\n" else 1
            item = self.line(self.buffer[:end])
            self.buffer = self.buffer[end + width:]
            if item is not None:
                events.append(item)
        if final:
            # EOF is not an event terminator: discard an incomplete final event.
            self.buffer, self.data = "", []
        return events


def encode_event(data, event=None, event_id=None, retry=None, comment=None, newline="\n"):
    """把一条事件编成 body 字节。空 data 且无其它字段时按注释处理。

    data 里的裸 CR / LF / CRLF 必须在这里切成多行：SSE 没有转义机制，
    留在字段值里的换行会被解码端当成行结束。代价是原始 CR 不保留，
    解码回来统一是 `\\n`——所以这里只保证「行结构」往返一致。
    """
    lines = []
    if comment is not None:
        for part in _LINES.split(comment):
            lines.append(f": {part}")
    if event_id is not None:
        assert "\x00" not in event_id
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        assert str(retry).isascii() and str(retry).isdecimal()
        lines.append(f"retry: {retry}")
    for part in ([] if data is None else _LINES.split(str(data))):
        lines.append(f"data: {part}")
    return (newline.join(lines) + newline + newline).encode("utf-8")


def self_test():
    raw = ("\ufeff: comment\r\nid: 7\r\nevent: delta\r\ndata: 中\r\ndata: 文\r\n\r\n"
           "retry: 1000\rdata: [DONE]\r\rdata: unfinished").encode()
    expected = [{"event": "delta", "data": "中\n文", "id": "7"},
                {"event": "message", "data": "[DONE]", "id": "7"}]
    for cut in range(len(raw) + 1):
        parser = SSEDecoder()
        actual = parser.feed(raw[:cut]) + parser.feed(raw[cut:], final=True)
        assert actual == expected, (cut, actual)
        assert parser.retry == 1000
    parser = SSEDecoder()
    actual = []
    for byte in raw:
        actual += parser.feed(bytes([byte]))
    actual += parser.feed(b"", final=True)
    assert actual == expected
    print(json.dumps({"split_positions_passed": len(raw) + 1, "one_byte_chunks": True,
                      "utf8_crlf_multiline_id_retry": True, "incomplete_eof_discarded": True,
                      "events": expected}, ensure_ascii=False))


def round_trip_test(rounds=2000, seed=511):
    """编码器写出来的字节，按任意切分喂回解码器必须还原成同一批事件。"""
    rng = random.Random(seed)
    alphabet = ["a", "中", "文", "\n", "\r", " ", ":", "", "🚀", "0"]
    checked = 0
    for _ in range(rounds):
        events = []
        for i in range(rng.randint(1, 4)):
            data = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
            if data == "":
                continue
            events.append({"data": data, "id": str(i)})
        if not events:
            continue
        body = b"".join(encode_event(e["data"], event_id=e["id"], newline=rng.choice(["\n", "\r", "\r\n"]))
                        for e in events)
        cuts = sorted(rng.sample(range(len(body) + 1), min(3, len(body) + 1)))
        parser = SSEDecoder()
        got = []
        prev = 0
        for cut in cuts:
            got += parser.feed(body[prev:cut])
            prev = cut
        got += parser.feed(body[prev:], final=True)
        want = [{"event": "message", "data": "\n".join(_LINES.split(e["data"])), "id": e["id"]}
                for e in events]
        assert got == want, (body, cuts, got, want)
        checked += 1
    print(json.dumps({"round_trip_cases": checked, "seed": seed,
                      "note": "data 内含 CR/LF/冒号与多字节字符；每个用例随机切 3 刀；"
                              "CR 被规范化为 LF，只比对行结构"},
                     ensure_ascii=False))


if __name__ == "__main__":
    self_test()
    round_trip_test()

