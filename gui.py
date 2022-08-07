import torch
import streamlit as st
import torch
from processed_input import ProcessedInput
from MLP import MultiLayerPerceptron

st.title("50.021 AI Project Gui")

model = MultiLayerPerceptron(in_dim=10001, out_dim=4)
model.load_state_dict(torch.load("mlp-model.pt", map_location=torch.device('cpu')))

headline = st.text_area("Please input the headline","")
article = st.text_area("Please input the article","")
print(f"headline: {headline}")
print(f"article: {article}")

if (headline != "") and (article != ""):
    processed_input = ProcessedInput(headline, article)
    feature = processed_input.get_feature()

    with torch.no_grad():
        feature = torch.tensor(feature).float()
        output = model(feature)
        output = output.argmax().item()

        number_to_stance = {
            0: "agree",
            1: "disagree",
            2: "discuss",
            3: "unrelated"
        }
        label = number_to_stance[int(output)]
        st.write(label)