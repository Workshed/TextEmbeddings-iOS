import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel
import coremltools as ct
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained model
model = AutoModel.from_pretrained('intfloat/e5-small')

class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super(NormalizedModel, self).__init__()
        self.base_model = base_model

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask):
        # Get the embeddings from the base model
        outputs = self.base_model(input_ids)
        last_hidden_state = outputs.last_hidden_state

        # Apply average pooling
        pooled_embeddings = self.average_pool(last_hidden_state, attention_mask)

        # Normalize the embeddings
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings

# Wrap your model with the normalization and pooling
normalized_model = NormalizedModel(model)

# Example input for tracing
example_input_ids = torch.randint(0, 10000, (1, 128))  # Adjust shape as needed
example_attention_mask = torch.ones((1, 128), dtype=torch.int64)  # Example attention mask

# Trace the model
traced_model = torch.jit.trace(normalized_model, (example_input_ids, example_attention_mask))

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=example_input_ids.shape),
        ct.TensorType(name="attention_mask", shape=example_attention_mask.shape)
    ],
)

# Save the CoreML model
mlmodel.save('normalized_pooled_model.mlpackage')


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small')

# Example sentence fragment
sentence_fragment = "The Manhattan bridge is"

# Tokenize the input sentence
# tokens = tokenizer(sentence_fragment, return_tensors="pt", padding=True, truncation=True)
tokens = tokenizer(
    sentence_fragment,
    return_tensors="pt",
    padding='max_length',  # Pad to the maximum length
    truncation=True,
    max_length=128  # Ensure the input length is 128
)

# Prepare the inputs for CoreML
coreml_inputs = {
    "input_ids": tokens['input_ids'].numpy().astype('int32'),
    "attention_mask": tokens['attention_mask'].numpy().astype('int32')
}

# Make predictions with the CoreML model
prediction_dict = mlmodel.predict(coreml_inputs)

# Print the keys of the prediction dictionary to inspect the output
print("Prediction keys:", prediction_dict.keys())

generated_tensor = prediction_dict["var_950"]

# Assuming generated_tensor is your normalized embedding from the model
embedding = generated_tensor.flatten()  # Flatten if necessary
print(f'Done: ${embedding}')
# # Ensure generated_tensor is a flat list of integers
# if isinstance(generated_tensor, list):
#     # Flatten the list if necessary
#     generated_tensor = [item for sublist in generated_tensor for item in sublist]

# # Decode the generated tensor
# generated_text = tokenizer.decode(generated_tensor)
# print("Fragment: {}".format(sentence_fragment))
# print("Completed: {}".format(generated_text))


# generated_text = tokenizer.decode(generated_tensor)
# print("Fragment: {}".format(sentence_fragment))
# print("Completed: {}".format(generated_text))

# # sentence_fragment = "The Manhattan bridge is"

# # tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small')
# # context = torch.tensor(tokenizer.encode(sentence_fragment))

# # coreml_inputs = {"context": context.to(torch.int32).numpy()}
# # prediction_dict = mlmodel.predict(coreml_inputs)
# # generated_tensor = prediction_dict["sentence_2"]
# # generated_text = tokenizer.decode(generated_tensor)
# # print("Fragment: {}".format(sentence_fragment))
# # print("Completed: {}".format(generated_text))