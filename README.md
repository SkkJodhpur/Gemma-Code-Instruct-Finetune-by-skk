---
library_name: transformers
tags:
  - healthcare
  - doctor-patient interaction
  - fine-tuning
  - LoRA
  - quantization
---


# Model

Fine-Tuning GEMMA-2B for Doctor-Patient Interaction: Efficient Model Adaptation Using LoRA and 4-bit Quantization

# Model Description
This model is a fine-tuned version of GEMMA-2B, adapted specifically for doctor-patient interaction tasks. Fine-tuning was performed using Low-Rank Adaptation (LoRA) and 4-bit quantization for efficient model adaptation. The model aims to facilitate improved, context-aware, and relevant interactions in medical consultations, enhancing communication and understanding between healthcare providers and patients.

# Developed by: Shailesh Kumar Khanchandani
# Shared by: Shailesh Kumar Khanchandani
# Model type: Causal Language Model
# Language(s) (NLP): English
# Finetuned from model: google/gemma-2b

# Model Sources
Repository: Fine-Tuned GEMMA-2B

# Uses
# Direct Use
The model is intended for direct use in generating contextually appropriate responses for doctor-patient interactions. It can be used in virtual assistants, chatbots, and other medical consultation platforms to assist healthcare providers in communicating with patients.

# Downstream Use
The model can be further fine-tuned for specific medical specialties or customized to fit the unique requirements of different healthcare systems or languages.

# Out-of-Scope Use
The model is not intended for use in generating medical advice or diagnoses without human oversight. It should not be used in any scenario where critical health decisions are made solely based on its outputs.

# Bias, Risks, and Limitations
The model inherits biases present in the training data. Users should be cautious of potential biases and inaccuracies, particularly in sensitive medical contexts.

# Recommendations
Users should ensure that outputs are reviewed by qualified healthcare professionals before being utilized in a clinical setting. Continuous monitoring and updating of the model with diverse and representative data can help mitigate biases.


# How to Get Started with the Model

Use the code below to get started with the model:

# python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_text(prompt):
    # Model and tokenizer initialization
    model_name = "skkjodhpur/Gemma-Code-Instruct-Finetune-by-skk"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

        # Tokenize input
    input_ids = tokenizer.encode(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
    # Example usage
    prompt = "Write a Python function to calculate the factorial of a number."
    response = generate_text(prompt)
    print("Generated response:")
    print(response)
