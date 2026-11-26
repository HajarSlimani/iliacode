import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)

paires = json.load(open("paires.json", encoding="utf-8"))

def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}],
        tokenize=False,
        add_generation_prompt=True)

    ids_demande = tokeniseur(demande)["input_ids"]
    ids = tokeniseur(
        demande + paire["reponse"] + "<|im_end|>"
    )["input_ids"]

    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]

    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


def entrainer():
    meilleure_perte = float("inf")
    meilleure_configuration = None

    for r in [4, 8, 16]:
        for learning_rate in [1e-4, 2e-4, 5e-4]:

            print(f"\nConfiguration : r={r}, lr={learning_rate}")

            moteur = AutoModelForCausalLM.from_pretrained(
                MOTEUR,
                torch_dtype=torch.bfloat16,
                device_map="cuda")

            moteur.gradient_checkpointing_enable()
            moteur.enable_input_require_grads()

            moteur = get_peft_model(
                moteur,
                LoraConfig(
                    r=r,
                    target_modules=["q_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM"))

            entraines = [
                p for p in moteur.parameters()
                if p.requires_grad
            ]

            for p in entraines:
                p.data = p.data.float()

            print(
                "paramètres entraînés",
                sum(p.numel() for p in entraines))

            adam = torch.optim.AdamW(
                entraines,
                lr=learning_rate)

            meilleure_perte_config = float("inf")
            compteur = 0

            random.seed(0)

            for epoque in range(20):
                random.shuffle(paires)
                pertes = []

                for paire in paires:
                    ids, cibles = exemple(paire)

                    perte = moteur(
                        input_ids=ids,
                        labels=cibles).loss

                    perte.backward()
                    adam.step()
                    adam.zero_grad()

                    pertes.append(perte.item())

                perte_moyenne = sum(pertes) / len(pertes)

                print(
                    f"époque {epoque + 1}, "
                    f"perte moyenne {perte_moyenne:.3f}")

                if perte_moyenne < meilleure_perte_config:
                    meilleure_perte_config = perte_moyenne
                    compteur = 0
                else:
                    compteur += 1

                if perte_moyenne < meilleure_perte:
                    meilleure_perte = perte_moyenne
                    meilleure_configuration = (r, learning_rate)

                    for p in entraines:
                        p.data = p.data.to(torch.float16)

                    moteur.save_pretrained(SORTIE)

                    for p in entraines:
                        p.data = p.data.float()

                    print("Meilleure configuration sauvegardée")

                if compteur >= 3:
                    print("Arrêt : la perte ne s'améliore plus")
                    break

            del moteur, adam
            torch.cuda.empty_cache()

    print("\nMeilleure configuration :", meilleure_configuration)
    print("Meilleure perte :", meilleure_perte)

    return meilleure_configuration, meilleure_perte


meilleur = entrainer()
print("Résultat final :", meilleur)