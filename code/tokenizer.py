import os

files = os.path.dirname(__file__) + "\\files\\"

class BasicTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size 
        self.vocab = None
        self.merges = None

    # <bpetraining>
    @staticmethod
    def get_stats(ids): # Get the most frequent pair in the byte-encoded text
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            return counts # Dictionary of the form "pair -> times_appeared"
    @staticmethod
    def merge(ids, pair, idx): # Update the token sequence according to a new merge
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, verbose=False):
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

        # --
        num_merges = self.vocab_size - 256
        ids = list(tokens) # copy so we don't destroy the original list

        merges = {} # (int, int) -> int
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
        self.merges= merges
        # <bpetraining>


    def encode(self,text):
        # given a string, return list of integers (the tokens)
        assert self.merges != None, "encode: First train your tokenizer!"
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def decode(self,tokens):
        assert self.merges != None, "decode: First train your tokenizer!"
        print(self.merges)
        return 
        
tokenizer = BasicTokenizer(1000)
with open(files + "taylorswift.txt", "r") as file:
    tokenizer.train(file.read(), verbose=False)


tokenized = tokenizer.encode("Hello")
tokenized_bits = list(map(int, tokenized))
dummy = "Hello".encode("utf-8")
dummy1 = list(map(int, dummy))
print(tokenized_bits)
print(dummy1)