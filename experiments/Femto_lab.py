import torchaudio
import argparse
"""
def custom_collate_func(batch):

    max_len = max(b. for b in batch)

    return max_len


x = torch.tensor([1,2,3,4])
y = torch.tensor([5,6,7,8,9])
z = torch.tensor([10,11,12,13,14])

batch = (x,y,z)


print(custom_collate_func(batch))"""

def inspect_audio(path):
    audio, sample_rate = torchaudio.load(path)
    print(audio.shape)
    print(sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an audio file with torchaudio.")
    parser.add_argument("path")
    inspect_audio(parser.parse_args().path)
