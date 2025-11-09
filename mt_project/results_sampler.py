import pandas as pd
from tqdm import tqdm
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizer #Mbart
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer #Byt5

#Funktion som tar filepath till fil med meningar och filepaths till alla modell-directories:
#1. Öppnar fil från filepath med meningar
#2. Öppnar outputfil och skriver kolumnrubriker
#3. Per mening i filen tokeniserar och ger meningen åt alla modeller samtidigt och skriver in meningen plus alla modellers output på rad i, med respektive modells output i dess egna kolumn
#3.1 strukturen blir:           original sentence | out1 | out2 | out3 | out4       med första raden som kolumnrubriker
#3.2 Flushar/closear filen efter varje rad för att spara medans man kör
#4. printar ett meddelande när allt är färdigt

class ModelRunner:
    """This class feeds all models of the project one sentence at a time and writes the output to a csv file with one column per model"""
    def __init__(self, data, mpath1, mpath2): #, mpath3):
        self.data = data
        self.mpath1 = mpath1
        self.mpath2 = mpath2
        #self.mpath3 = mpath3 #Ta bort comment sen
        #self.mpath4 = mpath4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #add instantiations of models and their tokenizers (calling load model 3 times and load moses once)
        self.cc25_model, self.cc25_tok = self.load_bart25(self.mpath1)
        self.mB50_model, self.mB50_tok = self.load_bart50(self.mpath2)
        #self.byt5_model, self.byt5_tok = self.load_byt5(self.mpath3)
        #self.moses_model = self.load_moses(self.mpath3) #Ta bort comment sen
        #print(self.data)
        print("model ", self.cc25_model, self.mB50_model)
    def translate(self):

        outdata = {"Original sentence Ukrainian": [],
                "Original sentence English": [],
                "mBART-Large-cc25": [],
                "mBART50": []
        } #This adds column names and structure of data

        with open(self.data, "r") as d: #idea is that in the data file, there is one sentence per line
            lines = d.readlines()
            for i, line in enumerate(tqdm(lines, desc="Translating sentences: ")):
                if i % 2 != 0: #Ukrainian if even, English if uneven
                #for line in tqdm(d, desc = "Translating sentences: "): #one line at a time
                    sentence = line.strip() #remove \n
                    print(sentence)
                    """Generate translations from each model"""
                    cc25_out = self.run_cc25(sentence)
                    mB50_out = self.run_mB50(sentence)
                    #byt5_out = self.run_byt5(sentence)
                    #moses_out = self.run_moses(sentence) #Ta bort comment sen
                    """Append translations from each model to corresponding column"""
                    outdata["Original sentence Ukrainian"].append(sentence) #Add Ukrainian
                    outdata["Original sentence English"].append(lines[i-1].strip()) #Add English (always before Ukrainian)
                    outdata["mBART-Large-cc25"].append(cc25_out)
                    outdata["mBART50"].append(mB50_out)
                    #outdata["Byt5"].append(byt5_out)
                    #outdata["Moses"].append(moses_out) #Ta bort comment sen
            
            #make dataframe, this indention
            df = pd.DataFrame.from_dict(outdata)
            df.to_csv("Translations_aspect.tsv", sep="\t", encoding = "utf-8", index = False) #.tsv to avoid , or ; in sentence

    def load_bart50(self, path):
        model = MBartForConditionalGeneration.from_pretrained(path)
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="uk_UA", tgt_lang="en_XX")
        tokenizer.src_lang = "uk_UA"
        tokenizer.tgt_lang = "en_XX"
        model = model.to(self.device)
        return model, tokenizer
    
    def load_bart25(self, path):
        model = MBartForConditionalGeneration.from_pretrained(path)
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        tokenizer.tgt_lang = "en_XX"
        tokenizer.src_lang = "ru_RU"
        model = model.to(self.device)
        return model, tokenizer


    def load_byt5(self, path): #removed from project
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model.to(self.device)
        return model, tokenizer
    
    def load_moses(self, path):
        model = _
        return model
            
    def run_cc25(self, sentence):
        inputs = self.cc25_tok(sentence, return_tensors = "pt").to(self.device)
        generated_tokens = self.cc25_model.generate(
            **inputs,
            forced_bos_token_id = self.cc25_tok.lang_code_to_id[self.cc25_tok.tgt_lang]
        )
        translation = self.cc25_tok.decode(generated_tokens[0], skip_special_tokens = True) #index 0 for translation
        return translation

    def run_mB50(self, sentence):
        inputs = self.mB50_tok(sentence, return_tensors = "pt").to(self.device)
        generated_tokens = self.mB50_model.generate(
            **inputs,
            forced_bos_token_id = self.mB50_tok.lang_code_to_id[self.mB50_tok.tgt_lang]
        )
        translation = self.mB50_tok.decode(generated_tokens[0], skip_special_tokens = True) #index 0 for translation
        return translation

    def run_byt5(self, sentence):
        inputs = self.byt5_tok(sentence, return_tensors = "pt").to(self.device)
        output = self.byt5_model.generate(**inputs)
        translation = self.byt5_tok.decode(output[0], skip_special_tokens = True)
        return translation

    def run_moses(self, sentence):
        return



#att tänka på
# .to("cuda") för GPU - KLART
#pandas eller csv direkt? - KLART (dataframe på slutet, se längs ned i listan)
#language codes för bert-modellerna (och hur fan kör man dem?) - KLART (använd mbart50 tokenizer för mbartcc25 också)
#Ska jag ha nån loading bar? D e ju nice - KLART
#print(f"Translated {i} out of {len(sen_file) sentences}) - KLART (loading bar istället)
#i steg 3 behöver nog modellerna kallas i separata funktioner som kallas i från huvudfunktionen och returnar översättningar - KLART
#ByT5 och mBARTs behöver olika tokenizers och kall-formuleringar - KLART
#Gör om till dataframe på slutet och skriv den till csv - KLART

def main():
    #Instantiate and run class with all file paths and data. Data can not be tokenized.
    #data = "/proj/uppmax2025-3-5/private/linus/project/clean_sentences.txt"
    data = "aspect_test_sentences.txt"
    mpath1 = "/proj/uppmax2025-3-5/private/jakob/project/output/checkpoint-81372"
    mpath2 = "/proj/uppmax2025-3-5/private/linus/project/output/checkpoint-78000"
    #mpath3 = byt5_path
    #mpath4 = moses_path
    print("main")
    runner = ModelRunner(data, mpath1, mpath2) #mpath3) #Lägg tillbaks mpath3 sen för MOSES #This order for models: cc25, mBERT50, byt5, moses
    runner.translate()

print("global")
main()
