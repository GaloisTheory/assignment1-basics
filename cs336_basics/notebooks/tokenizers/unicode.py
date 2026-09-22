# %%
chr(0)

char = "\x00"
char.__repr__()

print(char)

test1 = "This is a test" + chr(0) + "string"
print(test1)
# %%
test1
# %%
new_test = "\hello"

new_test.__repr__()

print(new_test)
# %%
test2 = ""
test2.__repr__()
# %%
ord("\x00")
# %%
test_string = "안녕하세요!adsf"

utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)

utf16_encoded = test_string.encode("utf-16")
print(utf16_encoded)

print(len(utf8_encoded))
print(len(utf16_encoded))

list(utf8_encoded)
# %%
list(utf16_encoded)


# %%
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))  # noqa: UP012

for word in ["hello", "안", "😀"]:
    print("The word is ", word)
    word_encoded = word.encode("utf-8")
    print(type(word_encoded))
    print(len(word_encoded))
    for b in word_encoded:
        print(b)
        print(bytes([b]))
        try:
            print(bytes([b]).decode("utf-8"))
        except UnicodeDecodeError:
            print("Error here")

# %%
print(word_encoded.hex(" "))  # Each two-digit group is one byte
print(word_encoded[:2].hex(" "))
for char in word:
    encoded = char.encode("utf-8")
    print(char, encoded.hex(" "), len(encoded))
# %%
