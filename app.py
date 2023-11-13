import streamlit as st
import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import PeftModel

# Tokenizer


@st.cache(allow_output_mutation=True)
def get_model():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')1
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    access_token = "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"
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,token=access_token,)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        token=access_token,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    model=PeftModel.from_pretrained(base_model,"ash100/llm-upload-test")
    return llama_tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    # test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # # test_sample
    # output = model(**test_sample)
    # st.write("Logits: ",output.logits)
    # y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    # st.write("Prediction: ",d[y_pred[0]])
    pmodel.eval()
    with torch.no_grad():
        ans=(llama_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=300, pad_token_id=2)[0], skip_special_tokens=True))
        st.write("Answer: "+ans)