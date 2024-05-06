
from transformers import AutoTokenizer, AutoModelForCausaLLM
import transformers

import torch

import json

print('finished imports')

model = "tiiuae/falcon-7b"
device = 'cuda'

if not torch.cuda.is_available():
    print('CUDA is not available, make sure you have GPU')
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model)

# loaded tokenizer

pipeline = transformers .pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
# torch_dtype=torch. floats,
# trust_remote_code=True,
    trust_remote_code=False,
    device=device,
)

partial_summary ="""

As previously disclosed, on April 23, 2023, Altitude Acquisition Corp., a Delaware corporation and a direct, wholly owned subsidiary of the Company, and HunniwelL Picard I, LLC entered into a business combination agreement (the “Picard Business Combination Agreement”) to merge with and into Vesicor Therapeutics, Inc., a California-based biotech company focused on microvesi
cle-based therapeutics. According to the Termination Agreement, the parties entered into mutual Termination and Release Agreement (the "Termination Agreement") to terminate the Picard BusinessCombination Agreement, which provides for a mutual release of claims among the parties and their affiliates, including, without Limitation, claims of fraud and willful breach. The T
ermination agreement also provides for the release of any claims that arise from fraud or willful breach of the agreement. In a parallel action, the U.S. Attorney's Office for the Southern District of New York today announced criminal charges against the defendants. A The SEC's complaint, filed in federal district court in New York, charges Altitude with violations of t
he antifraud provisions of Section 10(b) of the Securities Exchange Act of 1934 and Rule 10b-5 thereunder, and seeks a permanent injunction, disgorgement of ill-gotten gains with prejudgment interest, a civil penalty, and an officer-and-director bar. Without admitting or denying the allegations in the complaint, the defendants consented to the entry of a final judgment th
at permanently enjoins them from violating the charged provisions of the federal securities laws, and orders them to pay a termination fee as a result of the mutual decision to enter into the termination agreement. The settlement is subject to court approval. In addition, the court granted the SEC's request for an asset freeze and a temporary restraining order against Alt
itude and its related entities, as well as the appointment of an independent receiver to determine the amount of any civil penalty to be imposed by the court at a later date upon motion of the court. On May 2, 2020, the Court granted the court's motion for a preliminary injunction against further action by the defendants, and ordered that the court enter into a
"""

sequences = pipeline(
"provide a three or four sentence summary that captures the most important points of the following text: \n" + partial_summary,
# "What is 1 +12",
# max_Length=5000,
do_sample=True,

num_return_sequences=1
eos_token_id=tokenizer.eos_token_id,
>}
generated_text = sequences[0]['generated_text']
print (generated_text)

