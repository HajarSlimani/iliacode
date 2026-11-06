# Le choix du moteur d'ILIACode

## La machine

- GPU :
- mémoire G :          Go
- cache prévu : 4 demandes de 8 192 tokens, soit 32 768 tokens
- part fixe : 1,5 Go
- N_max :          Mds
- classe de moteur retenue :

## La grille

Une ligne par candidat. N, la mémoire 2N et l'étiquette de licence se
lisent sur la page du moteur, la licence complète dans son fichier LICENSE.
D/N et HumanEval viennent du cours.

| candidat | D/N | N | mémoire 2N | tient ? | licence, étiquette | licence, fichier LICENSE | HumanEval | verdict |
|---|---|---|---|---|---|---|---|---|
| mistralai/Codestral-22B-v0.1 | non publié | 22 | 44 | | mnpl |  non commerciale |  | écarté, taille |
| Qwen/Qwen2.5-Coder-3B | 7605 | 3.09 | 6.18 | | qwen-research | recherche seule | |écarté, licence |
| 01-ai/Yi-Coder-9B |  716|  9| 18|  | apache-2.0 |libre | 53.7 | classé 2e |
| Qwen/Qwen2.5-Coder-7B | 3092 |7.61  | 15.2 |  |  apache-2.0 | libre | 61.6 | retenu |
| Qwen/Qwen2.5-Coder-1.5B | 15260 | 1.54 | 3.08 |  | apache-2.0 | libre |  |témoin  |

## La décision

- moteur retenu : Qwen/Qwen2.5-Coder-7B
- témoin : Qwen/Qwen2.5-Coder-1.5B
- pourquoi, en trois phrases : Le modèle Qwen2.5-Coder-7B offre un bon compromis entre la taille, les performances en génération de code et les ressources matérielles disponibles. Sa licence Apache 2.0 permet une utilisation libre, y compris commerciale. Le modèle Qwen2.5-Coder-1.5B servira de témoin pour comparer les performances et la consommation des ressources avec celles du modèle retenu.

## La mesure interne

Bits par caractère du moteur retenu sur le test de la séance 2.

| bloc | bits par caractère |
|---|---|
| code |  |
| anglais |  |
| français |  |
| darija |  |
