import json
import math
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

# --- réglages (nouveau) ---
R = 16                                   # essais : 8, 16, 32
CIBLES = ["q_proj", "v_proj", "o_proj"]  # k_proj exclu
EPOQUES = 3
ACCUMULATION = 8
LR = 2e-4
MAX_TOKENS = 1024

random.seed(0)
torch.manual_seed(0)

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()

moteur = get_peft_model(moteur, LoraConfig(
    r=R, lora_alpha=2 * R, lora_dropout=0.05, target_modules=CIBLES, task_type="CAUSAL_LM"))
moteur.train()  # nouveau : from_pretrained renvoie un moteur en mode évaluation
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    # modifié : demande et réponse tokenisées séparément, frontière du masque sûre
    ids_demande = tokeniseur(demande, add_special_tokens=False)["input_ids"]
    ids_reponse = tokeniseur(paire["reponse"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(ids_demande) + len(ids_reponse) > MAX_TOKENS:
        return None
    ids = ids_demande + ids_reponse
    cibles = [-100] * len(ids_demande) + ids_reponse
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda(), len(ids_reponse)


paires = json.load(open("paires.json", encoding="utf-8"))
exemples = [e for e in (exemple(p) for p in paires) if e is not None]
print(f"{len(exemples)} exemples retenus sur {len(paires)} paires")

# nouveau : validation de 10 %, découpage fixe pour comparer les essais
random.Random(1234).shuffle(exemples)
n_val = max(1, len(exemples) // 10)
validation, entrainement = exemples[:n_val], exemples[n_val:]


def perte_validation():
    moteur.eval()
    somme, n = 0.0, 0
    with torch.no_grad():
        for ids, cibles, n_rep in validation:
            somme += moteur(input_ids=ids, labels=cibles).loss.item() * n_rep
            n += n_rep
    moteur.train()
    return somme / n


pas_total = math.ceil(len(entrainement) / ACCUMULATION) * EPOQUES
adam = torch.optim.AdamW(entraines, lr=LR)
planning = get_cosine_schedule_with_warmup(adam, int(0.05 * pas_total), pas_total)  # nouveau

print(f"perte de validation avant affinage {perte_validation():.3f}")
meilleure_val, meilleur_etat = float("inf"), None
for epoque in range(EPOQUES):
    random.shuffle(entrainement)
    pertes, n_tokens = 0.0, 0
    for debut in range(0, len(entrainement), ACCUMULATION):
        groupe = entrainement[debut:debut + ACCUMULATION]
        n_groupe = sum(n for _, _, n in groupe)
        for ids, cibles, n in groupe:
            perte = moteur(input_ids=ids, labels=cibles).loss
            (perte * n / n_groupe).backward()  # nouveau : accumulation pondérée par token
            pertes += perte.item() * n
            n_tokens += n
        torch.nn.utils.clip_grad_norm_(entraines, 1.0)  # nouveau
        adam.step()
        planning.step()
        adam.zero_grad()
    val = perte_validation()
    print(f"époque {epoque + 1}, perte entraînement {pertes / n_tokens:.3f}, perte validation {val:.3f}")
    if val < meilleure_val:  # nouveau : on garde la meilleure époque
        meilleure_val = val
        meilleur_etat = [p.detach().cpu().clone() for p in entraines]

with torch.no_grad():
    for p, sauve in zip(entraines, meilleur_etat):
        p.copy_(sauve.to(p.device))
for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)
