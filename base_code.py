# Importeer libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from scipy import stats

# Lees de CSV file en noem het dataframe df
# nrows=44 om niet meer dan 44 datapunten in het dataframe te zetten
# Vervolgens printen we de dataset.
df = pd.read_csv('metingen.csv', header=0, nrows=44)


print("Dit zijn de aardappel massa's in gram:")
print("\n")
print(df)

# Bereken totaal aantal metingen, minimale en maximale massa
# Gebruik len om het totaal aantal metingen te berekenen
# Gebruik min en max om de minimale en maximale massa te berekenen
totaal_metingen = len(df)
min_massa = df['massa'].min()
max_massa = df['massa'].max()

# Print de resultaten
print(f'Minimale massa: {min_massa}')
print(f'Maximale massa: {max_massa}')
print(f'Totaal aantal metingen: {totaal_metingen}')


bins = math.ceil(np.sqrt(totaal_metingen))
print(f'Aantal klasse-intervallen: {bins}')
# linspace genereert een array van waarden tussen min_massa en max_massa, inclusief de grenzen
klassengrenzen = np.linspace(min_massa, max_massa, bins + 1)
# pd.cut maakt klasse-intervallen op basis van de klassengrenzen, include_lowest=True zorgt ervoor dat de laagste waarde in de eerste klasse valt, right=True zorgt ervoor dat de rechtergrens van de klasse wordt inbegrepen in het interval
df['klasse'] = pd.cut(df['massa'], bins=klassengrenzen, include_lowest=True, right=True)
# Print de breedte van de klassen
klasse_breedtes = df['klasse'].apply(lambda x: x.right - x.left)
print("\nBreedte van de klassen (alleen unieke waarden):")
print(klasse_breedtes.unique())
# 1. Absolute frequentie
# Hoe vaak komt elke klasse voor in de dataset
absolute_frequentie = df['klasse'].value_counts().sort_index()
# 2. Relatieve frequentie
# Relatieve frequentie is de absolute frequentie gedeeld door het totaal aantal metingen
relatieve_frequentie = absolute_frequentie / totaal_metingen
# 3. Cumulatieve frequentie
# Cumulatieve frequentie is de som van de absolute frequenties tot en met de huidige klasse, cumsum() berekent de cumulatieve som
cumulatieve_frequentie = absolute_frequentie.cumsum()

print("\nAbsolute frequentie:")
print(absolute_frequentie)
print("\nRelatieve frequentie:")
print(relatieve_frequentie)
print("\nCumulatieve frequentie:")
print(cumulatieve_frequentie)

# Bepaal het gemiddelde van de massa op basis van losse metingen
gemiddelde_massa_los = (df['massa'].sum() / totaal_metingen) if totaal_metingen > 0 else 0
print(f'\n--- Deel 1: Op basis van losse metingen ---')
print(f'Gemiddelde massa (losse metingen): {gemiddelde_massa_los:.2f} g')
# De gebruikte formule
print("Formule gemiddelde (losse metingen): x̄ = Σxi / n")

# Bepaal de steekproef standaardafwijking van de massa (losse metingen)
print("\nBerekening steekproefstandaardafwijking (losse metingen):")
print("Formule steekproefvariantie (s²): Σ(xi - x̄)² / (n-1)")
print("Formule steekproefstandaardafwijking (s): √s²")

# Stap 1: Bereken de gekwadrateerde afwijkingen van de massa (losse metingen): (xi - x̄_los)²
# xi is de massa van elke losse meting, x̄_los is het gemiddelde van de losse metingen
gekwadrateerde_afwijkingen_los = []
for xi in df['massa']:
    afwijking_los = xi - gemiddelde_massa_los
    gekwadrateerde_afwijkingen_los.append(afwijking_los ** 2)
    
    
# Stap 2: Sommeer de gekwadrateerde afwijkingen (losse metingen): Σ(xi - x̄_los)²
som_gekwadrateerde_afwijkingen_los = sum(gekwadrateerde_afwijkingen_los)
print(f'\nSom van gekwadrateerde afwijkingen Σ(xi - x̄_los)²: {som_gekwadrateerde_afwijkingen_los:.2f}')

# Stap 3: Bereken de steekproefvariantie s² (losse metingen) = Σ(xi - x̄_los)² / (n-1)
if totaal_metingen > 1:
    steekproef_variantie_los = som_gekwadrateerde_afwijkingen_los / (totaal_metingen - 1)
else:
    steekproef_variantie_los = 0 
print(f'Steekproefvariantie s² (losse metingen): {steekproef_variantie_los:.2f}')

# Stap 4: Bereken de steekproefstandaardafwijking s (losse metingen) = √s²_los
if steekproef_variantie_los >= 0:
    handmatige_standaardafwijking_los = math.sqrt(steekproef_variantie_los)
else:
    handmatige_standaardafwijking_los = 0
print(f'Steekproefstandaardafwijking s (losse metingen, handmatig): {handmatige_standaardafwijking_los:.2f} g')

# Ter controle met pandas .std() functie (losse metingen):
pandas_standaardafwijking_los = df['massa'].std(ddof=1)  # ddof=1 voor steekproef standaardafwijking
print(f'Pandas steekproefstandaardafwijking s (losse metingen, pandas): {pandas_standaardafwijking_los:.2f} g')

### Op basis van de frequentieverdeling
print("\n--- Deel 2: Op basis van frequentieverdeling ---")

# Stap 1: Bepaal de klasse-middens
# Voor elk interval (klasse), set de onder- en bovengrens en bereken het klasse-midden (gemiddelde van de onder- en bovengrens)
klasse_middens = []
for interval in absolute_frequentie.index:
    ondergrens = interval.left
    bovengrens = interval.right
    klasse_midden = (ondergrens + bovengrens) / 2
    klasse_middens.append(klasse_midden)
    
# Converteer naar numpy array voor verdere berekeningen
klasse_middens_np = np.array(klasse_middens)

print("Stap 1: Klasse-middens:")
# Weergave voor klasse-middens
frequentietabel_middens = pd.DataFrame({
    'Klasse': absolute_frequentie.index,
    'Klassemidden (mᵢ)': klasse_middens_np
})
print(frequentietabel_middens.to_string())
print(f"\nKlassemiddens als lijst: {klasse_middens_np.tolist()}")

# Stap 2: Bepaal het gemiddelde (x̄_freq) op basis van de frequentieverdeling
print("\n\nStap 2: Bereken het gemiddelde (x̄_freq) op basis van de frequentieverdeling")
print("Formule: x̄_freq ≈ Σ(fᵢ * mᵢ) / n")
print("Waarbij:")
print("fᵢ = absolute frequentie van klasse i")
print("mᵢ = klassemidden van klasse i")
print("n = totaal aantal metingen")

# 2a: Bereken fᵢ * mᵢ voor elke klasse
# absolute_frequentie.values bevat de fᵢ waarden
# klasse_middens_np bevat de mᵢ waarden
product_fi_mi = absolute_frequentie.values * klasse_middens_np
print(f"Product fᵢ * mᵢ: {product_fi_mi}")
# Presenteer data in een DataFrame
tabel_fi_mi = pd.DataFrame({
    'Klasse': absolute_frequentie.index,
    'fᵢ': absolute_frequentie.values,
    'mᵢ': klasse_middens_np,
    'fᵢ * mᵢ': product_fi_mi
})
print("\nTabel met absolute frequentie(fᵢ), klassenmidden(mᵢ) en fᵢ * mᵢ:")
print(tabel_fi_mi.to_string(index=False))

# 2b: Sommeer de producten (Σ(fᵢ * mᵢ))
som_product_fi_mi = product_fi_mi.sum()
print(f"\nSom van producten Σ(fᵢ * mᵢ): {som_product_fi_mi:.2f}")

# 2c: Deel door het totaal aantal metingen (n) om het gemiddelde te berekenen
if totaal_metingen > 0:
    gemiddelde_massa_freq = som_product_fi_mi / totaal_metingen
    print(f"Totaal aantal metingen (n): {totaal_metingen}")
    print(f"Benaderd gemiddelde (x̄_freq) = {som_product_fi_mi:.4f}/{totaal_metingen} = {gemiddelde_massa_freq:.2f}g")
else:
    gemiddelde_massa_freq = 0 # Voorkomen van deling door nul
    print("Totaal aantal metingen is 0, kan geen gemiddelde berekenen.")    
    
# Stap 3: Bereken de (benaderde) steekproef standaardafwijking (s_freq) op basis van de frequentieverdeling
# Formule: s_freq ≈ √(Σ(fᵢ * (mᵢ - x̄_freq)²) / (n-1))
# Waarbij:
# fᵢ = absolute frequentie van klasse i
# mᵢ = klassemidden van klasse i
# x̄_freq = gemiddelde op basis van frequentieverdeling
# n = totaal aantal metingen -> n-1 voor steekproefvariantie

print("\n\nStap 3: Bereken de (benaderde) steekproef standaardafwijking (s_freq) op basis van de frequentieverdeling")
print("Formule: s_freq ≈ √(Σ(fᵢ * (mᵢ - x̄_freq)²) / (n-1))")
print("Formule steekproefstandaardafwijking: s_freq = √s²_freq")


# We starten een iteratie om voor elke klasse de afwijking van het klassemidden tot het gemiddelde te berekenen en de variantie te bepalen.
# Controleer of het gemiddelde is berekend en n > 1 is. 
if totaal_metingen > 1 and 'gemiddelde_massa_freq' in locals() and gemiddelde_massa_freq is not None:
    # 3a: Bereken de afwijking can elk klassemidden tot het gemiddelde: (mᵢ - x̄_freq)
    afwijking_midden_vs_gemiddelde_freq = klasse_middens_np - gemiddelde_massa_freq
    
    # 3b: Kwadrateer de afwijkingen: (mᵢ - x̄_freq)²
    gekwadrateerde_afwijkingen_freq = afwijking_midden_vs_gemiddelde_freq ** 2

    # 3c: Vermenigvuldig met de frequentie van de klasse: fᵢ * (mᵢ - x̄_freq)²
    product_fi_gekwadrateerde_afwijkingen = absolute_frequentie.values * gekwadrateerde_afwijkingen_freq
    
    # Toon tussenstappen in een DataFrame
    tabel_variantie_stappen = pd.DataFrame({
        'Klasse': absolute_frequentie.index,
        'fᵢ': absolute_frequentie.values,
        'mᵢ': klasse_middens_np,
        'mᵢ - x̄_freq': afwijking_midden_vs_gemiddelde_freq,
        '(mᵢ - x̄_freq)²': gekwadrateerde_afwijkingen_freq,
        'fᵢ * (mᵢ - x̄_freq)²': product_fi_gekwadrateerde_afwijkingen
    })
    print("\nTabel met stappen voor berekening van steekproefvariantie:")
    print(tabel_variantie_stappen.to_string(index=False))
    
    # 3d: Sommeer de producten (Σ(fᵢ * (mᵢ - x̄_freq)²))
    som_product_fi_gekwadrateerde_afwijkingen = product_fi_gekwadrateerde_afwijkingen.sum()
    print(f"\nSom van producten Σ(fᵢ * (mᵢ - x̄_freq)²): {som_product_fi_gekwadrateerde_afwijkingen:.2f}")
    
    #3e: Bereken de steekproefvariantie s²_freq
    # De noemer is (n-1) voor steekproefvariantie
    steekproef_variantie_freq = som_product_fi_gekwadrateerde_afwijkingen / (totaal_metingen - 1)
    print(f"Steekproefvariantie s²_freq: {steekproef_variantie_freq:.2f}")
    
    # 3f: Bereken de steekproefstandaardafwijking s_freq
    if steekproef_variantie_freq >= 0:
        handmatige_standaardafwijking_freq = math.sqrt(steekproef_variantie_freq)
        print(f"Steekproefstandaardafwijking s_freq (handmatig): {handmatige_standaardafwijking_freq:.2f} g")
    else:
        handmatige_standaardafwijking_freq = 0
        print("\nStandaardafwijking (frequentieverdeling) kan niet worden berekend (onvoldoende metingen of gemiddelde niet beschikbaar).")

print("\n\n--- Vraag 3: Normal Probability Plot ---")

from scipy import stats

# Formaat van de plot
plt.figure(figsize=(10, 8))

# Stap 1: Bereken de data voor de pplot
# osm = Ordered Statistic Method (geobserveerde waarden, gesorteerd)
# osr = Ordered Statistic Rank (theoretische waarden / z-scores voor normale verdeling)
# De regressie lijn berekend door probplot is: osm = slope *osr + intercept
(osm, osr), (slope, intercept, r_value) = stats.probplot(df['massa'], dist="norm", plot=None)

# Stap 2: Maak de scatter plot
plt.scatter(osr, osm, label='Geobserveerde waarden (osm)', s=50)

# Stap 3: Voeg de regressielijn toe
# De oorspronkelijke lijn is: osm = slope * osr + intercept
# We willen nu osr and een functie van osm: osr= (osm - intercept) / slope
x_line_plot = np.linspace(osr.min(), osr.max(), 100)

if abs(slope) > 1e-9:  # Voorkom deling door nul
    y_line_plot = (x_line_plot - intercept) / slope
    plt.plot(x_line_plot, y_line_plot, 'r', label='Regressielijn (kleinste kwadraten)', linewidth=2)
else:
    # Dit scenario (slope is bijna nul) is onwaarschijnlijk voor typische data in een probplot.
    # Het zou betekenen dat de geobserveerde waarden nauwelijks variëren met de theoretische kwantielen.
    print("Waarschuwing: Helling van de regressielijn is zeer klein, de lijn kan mogelijk niet correct worden weergegeven.")

plt.title('Normal Probability Plot van Aardappel Massa (assen gedraaid)')
plt.xlabel('Geobserveerde Waarden (Massa g)') # Nu osm
plt.ylabel('Theoretische Waarden (z-scores)') # Nu osr
plt.grid(True)

# Toon R-kwadraat op de plot
r_squared = r_value**2
plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.legend()
plt.savefig('aardappels_normal_probability_plot_gedraaid.png') # Sla op met een nieuwe naam eventueel
plt.show()

# 4. Maak en toon de tabel met corresponderende data
# osm (geobserveerde waarden) en osr (theoretische waarden) zijn al gesorteerd zoals ze in de plot gebruikt worden.
data_tabel = pd.DataFrame({
    'Geobserveerde Waarde (Massa g)': osm,
    'Theoretische Waarde (z-score)': osr
})
print("\nTabel met data voor Normal Probability Plot:")
# .round(4) voor betere leesbaarheid van de waarden
print(data_tabel.round(4))

# Deze parameters horen bij de formule: geobserveerd = slope * theoretisch + intercept
print(f"\nParameters van de oorspronkelijke regressielijn (geobserveerd = slope * theoretisch + intercept):")
print(f"  Helling (slope): {slope:.4f}")
print(f"  Onderschepping (intercept): {intercept:.4f}")
print(f"  Correlatiecoëfficiënt (r-value): {r_value:.4f}")
print(f"  R-kwadraat: {r_squared:.4f}")

# Bevindingen
print("\n--- Bevindingen ---")
print("1. De gemiddelde massa van de aardappels is {:.2f} g op basis van losse metingen en {:.2f} g op basis van de frequentieverdeling.".format(gemiddelde_massa_los, gemiddelde_massa_freq))
print("2. De steekproef standaardafwijking is {:.2f} g op basis van losse metingen en {:.2f} g op basis van de frequentieverdeling.".format(handmatige_standaardafwijking_los, handmatige_standaardafwijking_freq))

print("\n\n--- Opdracht 4: Betrouwbaarheidsinterval van de verwachting μ ---")

# Bereken het 95% betrouwbaarheidsinterval voor het gemiddelde
# Formule: x̄ ± t_(α/2, n-1) * (s/√n)
alpha = 0.05  # Voor 95% betrouwbaarheidsinterval
n = totaal_metingen
df_t = n - 1  # vrijheidsgraden voor t-verdeling
# stats.t.ppf geeft de kritieke t-waarde voor een gegeven alpha en vrijheidsgraden
# Voor een tweezijdig betrouwbaarheidsinterval gebruiken we 1 - alpha/2
# df_t = n - 1  # Vrijheidsgraden voor t-verdeling
# Stats heeft geen tabellen, maar we kunnen de kritieke t-waarde berekenen met de t-verdeling
# de t-waarde wordt berekend door de percent-point function (ppf) van de t-verdeling
t_kritiek = stats.t.ppf(1 - alpha / 2, df_t)
print(f"Steekproefgemiddelde (x̄): {gemiddelde_massa_los:.2f} g")
print(f"Steekproefstandaardafwijking (s): {handmatige_standaardafwijking_los:.2f} g")
print(f"Steekproefgrootte (n): {n}")
print(f"Vrijheidsgraden (n-1): {df_t}")
print(f"Kritieke t-waarde (t_(α/2, n-1)) voor {(1-alpha)*100:.0f}% interval: {t_kritiek:.4f}")

# Standaardfout van het gemiddelde
standaardfout = handmatige_standaardafwijking_los / math.sqrt(n)
print(f"Standaardfout van het gemiddelde (s/√n): {standaardfout:.4f}")

# Bereken de foutmarge
foutmarge = t_kritiek * standaardfout
print(f"Foutmarge (t_(α/2, n-1) * (s/√n)): {foutmarge:.4f}")

# Bereken het betrouwbaarheidsinterval
ondergrens = gemiddelde_massa_los - foutmarge
bovengrens = gemiddelde_massa_los + foutmarge
print(f"\n{(1-alpha)*100:.0f}% Betrouwbaarheidsinterval: [{ondergrens:.2f}, {bovengrens:.2f}] g")

# Conclusies
print("\n--- Bevindingen ---")
print(f"1. De steekproef van {n} aardappelen heeft een gemiddelde massa van {gemiddelde_massa_los:.2f} g.")
print(f"2. Met 95% betrouwbaarheid kunnen we stellen dat het populatiegemiddelde")
print(f"   van de aardappelmassa's ligt tussen {ondergrens:.2f} g en {bovengrens:.2f} g.")
print(f"3. De standaardafwijking bedraagt {handmatige_standaardafwijking_los:.2f} g")

print("\n--- Betrouwbaarheidsinterval voor de Standaarddeviatie ---")

# Het berekenen van een betrouwbaarheidsinterval voor de standaarddeviatie
# maakt gebruik van de chi-kwadraat verdeling
alpha = 0.05  # Voor 95% betrouwbaarheidsinterval
n = totaal_metingen
df_chi2 = n - 1  # vrijheidsgraden voor chi-kwadraat verdeling

# Kritieke waarden van chi-kwadraat voor onder- en bovengrens
# Deze waarden worden berekend met de percent-point function (ppf) van de chi-kwadraat verdeling
# Dit doen we met de scipy.stats.chi2 module
# chi2.ppf geeft de kritieke waarde voor een gegeven alpha en vrijheidsgraden
# Voor een tweezijdig betrouwbaarheidsinterval gebruiken we alpha/2 voor de ondergrens en 1 - alpha/2 voor de bovengrens
chi2_lower = stats.chi2.ppf(alpha/2, df_chi2)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df_chi2)

print(f"Steekproefstandaarddeviatie (s): {handmatige_standaardafwijking_los:.4f} g")
print(f"Steekproefvariantie (s²): {steekproef_variantie_los:.4f}")
print(f"Vrijheidsgraden (n-1): {df_chi2}")
print(f"Chi-kwadraat ondergrens (χ²_(α/2, n-1)): {chi2_lower:.4f}")
print(f"Chi-kwadraat bovengrens (χ²_(1-α/2, n-1)): {chi2_upper:.4f}")

# Formule: √[(n-1)s² / χ²_(α/2, n-1)] < σ < √[(n-1)s² / χ²_(1-α/2, n-1)]
# Berekent de onder- en bovengrens van het betrouwbaarheidsinterval 
sd_lower = math.sqrt((df_chi2 * steekproef_variantie_los) / chi2_upper)
sd_upper = math.sqrt((df_chi2 * steekproef_variantie_los) / chi2_lower)

print(f"\nBerekening ondergrens: √[({df_chi2} × {steekproef_variantie_los:.4f}) / {chi2_upper:.4f}] = {sd_lower:.4f}")
print(f"Berekening bovengrens: √[({df_chi2} × {steekproef_variantie_los:.4f}) / {chi2_lower:.4f}] = {sd_upper:.4f}")

print(f"\n{(1-alpha)*100:.0f}% Betrouwbaarheidsinterval voor σ: [{sd_lower:.2f}, {sd_upper:.2f}] g")


print("\n--- Bevindingen ---")
print(f"De populatie standaardafwijking (σ) van de aardappelmassa's ")
print(f"ligt tussen {sd_lower:.2f} g en {sd_upper:.2f} g.")
print(f"Dit betekent dat de werkelijke variabiliteit in aardappelmassa's kan")
print(f"afwijken van de steekproefschatting ({handmatige_standaardafwijking_los:.2f} g), maar met 95%")
print(f"zekerheid binnen deze grenzen valt.")

print("\n\n--- Opdracht 5: Kans op Afwijking van het Gemiddelde ---")

# We gaan ervan uit dat de aardappelgewichten normaal verdeeld zijn
# Voor deze vraag beschouwen we de steekproef als populatie
# We gebruiken het gemiddelde en de standaardafwijking uit de losse metingen

print("We beschouwen de steekproef nu als populatie met:")
print(f"- Gemiddelde (μ): {gemiddelde_massa_los:.2f} g")
print(f"- Standaardafwijking (σ): {handmatige_standaardafwijking_los:.2f} g")

# We berekenen de kans dat een willekeurige aardappel meer dan 1,8 standaarddeviaties 
# van het gemiddelde af zit (boven en ondergrens)
z_score = 1.8

# Grenswaarden voor de z-score 
# Formule: x = μ ± z * σ
ondergrens_z = gemiddelde_massa_los - z_score * handmatige_standaardafwijking_los
bovengrens_z = gemiddelde_massa_los + z_score * handmatige_standaardafwijking_los

print(f"\nGrenswaarden voor ±{z_score} standaarddeviaties van het gemiddelde:")
print(f"- Ondergrens: {ondergrens_z:.2f} g")
print(f"- Bovengrens: {bovengrens_z:.2f} g")


# stats.norm.cdf geeft de cumulatieve distributiefunctie (CDF) van de normale verdeling
# Dit is een mathematische functie die een kans percentage geeft voor een bepaalde z-score
# We berekenen de kans dat een aardappel lichter is dan de ondergrens en zwaarder dan de bovengrens

# De ondergrens
# P(Z < -1,8)
kans_onder = stats.norm.cdf(-z_score)
print(f"\nKans dat een aardappel lichter is dan {ondergrens_z:.2f} g: {kans_onder:.4f} = {kans_onder*100:.2f}%")

# De bovengrens
# P(Z > 1,8)
kans_boven = 1 - stats.norm.cdf(z_score)
print(f"Kans dat een aardappel zwaarder is dan {bovengrens_z:.2f} g: {kans_boven:.4f} = {kans_boven*100:.2f}%")

# Totale kans P(|Z| > 1,8) = P(Z < -1,8) + P(Z > 1,8)
kans_totaal = kans_onder + kans_boven
print(f"\nTotale kans dat een willekeurige aardappel meer dan {z_score} standaarddeviaties")
print(f"van het gemiddelde afwijkt: {kans_totaal:.4f} = {kans_totaal*100:.2f}%")
