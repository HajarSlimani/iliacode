from bonjour import saluer
def test_saluer():
    assert saluer("Hajar") == "Bonjour Hajar"
def test_saluer_vide():
    assert saluer("") == "Bonjour "