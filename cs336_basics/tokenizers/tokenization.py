from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from io import BufferedReader
import regex as re 

from cs336_basics.tokenizers.pretokenization_example import find_chunk_boundaries 
from cs336_basics.tokenizers.config import DATA_PATH_TOY, DATA_PATH_TRAIN, DATA_PATH_VALID


def pretokenize(input_path: str, 
                special_tokens: list[str], 
                ) -> dict[tuple[bytes], int]: 
    with open(input_path, "r", encoding="utf-8") as f:
        #Pre-Tokenization 
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # split_text = re.findall(PAT, f.read())
        st_split_pattern = "|".join(re.escape(s) for s in special_tokens)
        chunks = re.split(st_split_pattern, f.read())
        
        pretokens = defaultdict(int)
        for chunk in chunks: 
            for match in re.finditer(PAT, chunk): 
                piece = match.group().encode("utf-8")
                piece_tuple = tuple(piece[i:i+1] for i in range(len(piece)))
                pretokens[piece_tuple] += 1

    return pretokens


def pretokenize_chunked(
    num_processes: int, 
    input_path: str, 
    special_tokens: list[str]
) -> dict[tuple[bytes], int]: 
    with open(input_path, "rb") as f:
        st_split_pattern = "|".join(re.escape(s) for s in special_tokens)
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
    with ProcessPoolExecutor(max_workers=num_processes) as pool: 
        results = list(
            pool.map(pretokenize_chunk, [[input_path, 
                                          boundaries[i], 
                                          boundaries[i+1], 
                                          st_split_pattern] for i in range(len(boundaries)-1)])
        )
    combined = Counter()
    for chunk_counts in results:
        combined.update(chunk_counts)
    
    return combined 


def pretokenize_chunk(
    args
) -> dict[tuple[bytes], int]: 
    input_path, start, end, st_split_pattern = args
    with open(input_path, "rb") as f:
        f.seek(start)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        chunks = re.split(st_split_pattern, f.read(end - start).decode("utf-8", errors="ignore"))

    pretokens = defaultdict(int)
    for chunk in chunks: 
        for match in re.finditer(PAT, chunk): 
            piece = match.group().encode("utf-8")
            piece_tuple = tuple(piece[i:i+1] for i in range(len(piece)))
            pretokens[piece_tuple] += 1

    return pretokens


def merge(
    pretokens: dict[tuple[bytes], int],
    new_vocab_size: int, 
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 

    vocab = {i: bytes([i]) for i in range(256)}  
    current_vocab_size = 256
    merge_list = []

    pairs = defaultdict(int)
    pair_to_pretoken = defaultdict(set)
    for pretoken, freq in pretokens.items(): 
        for a, b in zip(pretoken, pretoken[1:]): 
            pairs[(a, b)] += freq
            pair_to_pretoken[(a, b)].add(pretoken)

    while current_vocab_size < new_vocab_size: 
        max_pair = None
        max_freq = 0
        for pair, freq in pairs.items(): 
            if max_pair is None or (freq, pair) > (max_freq, max_pair): 
                max_pair = pair 
                max_freq = freq

        if max_pair is None: 
            break 

        merge_list.append(max_pair)
        vocab[current_vocab_size] = bytes(max_pair[0]+max_pair[1])
        current_vocab_size += 1
        
        #Now we update the necessary dictionaries 
        pretokens_to_check = pair_to_pretoken[max_pair].copy()
        for pretoken in pretokens_to_check: 
            n = len(pretoken)
            new = []
            freq = pretokens[pretoken]

            i = 0
            while i < n:
                if i < n-1 and (pretoken[i], pretoken[i+1]) == max_pair: 
                    new.append(pretoken[i] + pretoken[i+1])
                    i += 2
                    continue
                else: 
                    new.append(pretoken[i])
                i += 1
            
            new_pretoken = tuple(new)
            pretokens[new_pretoken] = freq

            for a, b in zip(pretoken, pretoken[1:]): 
                pairs[(a, b)] -= freq
                if pretoken in pair_to_pretoken[(a, b)]: 
                    pair_to_pretoken[(a, b)].remove(pretoken)

            for a, b in zip(new_pretoken, new_pretoken[1:]): 
                pairs[(a, b)] += freq
                pair_to_pretoken[(a, b)].add(new_pretoken)

            del pretokens[pretoken]

        del pairs[max_pair]
        del pair_to_pretoken[max_pair]

    return vocab, merge_list
    

def train_bpe(
    input_path: str, 
    vocab_size: int = 262, 
    special_tokens: list[str] = ["<|endoftext|>"], 
    num_processes: int = 4, 
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    if num_processes == 1: 
        pretokens = pretokenize(input_path=input_path, 
                                special_tokens=special_tokens)
    else: 
        pretokens = pretokenize_chunked(num_processes=num_processes, 
                                        input_path=input_path, 
                                        special_tokens=special_tokens)

    vocab, merge_list = merge(pretokens=pretokens, 
                              new_vocab_size=vocab_size-len(special_tokens))

    current_vocab_size = len(vocab)
    for special_token in special_tokens:
        vocab[current_vocab_size] = special_token.encode("utf-8")
        current_vocab_size += 1

    return vocab, merge_list


if __name__ == '__main__': 
    vocab_valid, merge_list_valid = train_bpe(DATA_PATH_VALID, vocab_size=4096)

    print(merge_list_valid)