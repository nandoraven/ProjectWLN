# Importeer libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from scipy import stats
import streamlit as st
import io

# App configuratie
st.set_page_config(
    page_title="Basis Statistiek Tool",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App titel en inleiding
st.title("Basis Statistiek Tool")
st.markdown("""
Deze app helpt je met statistische analyse van gegevens, specifiek gericht op aardappelmassa's.
Upload je eigen dataset of gebruik de voorbeeld dataset om analyses uit te voeren.
""")

# Sidebar voor bestandsupload en configuratie
st.sidebar.header("Instellingen")

# Optie om een bestand te uploaden of het voorbeeldbestand te gebruiken
use_example = st.sidebar.checkbox("Gebruik voorbeeld dataset", value=True)

if use_example:
    # Gebruik het voorbeeldbestand
    df = pd.read_csv('metingen.csv', header=0)
    st.sidebar.info("Voorbeeld dataset geladen")
else:
    # Bestand upload mogelijkheid
    uploaded_file = st.sidebar.file_uploader("Upload je CSV bestand", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=0)
        st.sidebar.success("Bestand succesvol geüpload!")
    else:
        st.sidebar.warning("Geen bestand geüpload, gebruik het voorbeeldbestand")
        df = pd.read_csv('metingen.csv', header=0)

# Toon de dataset
st.header("Dataset")
st.write("Dit zijn de aardappel massa's in gram:")
st.dataframe(df)

# Bereken totaal aantal metingen, minimale en maximale massa
# Gebruik len om het totaal aantal metingen te berekenen
# Gebruik min en max om de minimale en maximale massa te berekenen
totaal_metingen = len(df)
min_massa = df['massa'].min()
max_massa = df['massa'].max()

# Toon de resultaten in een expander
with st.expander("Basisinformatie over de dataset", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Minimale massa", f"{min_massa} g")
    col2.metric("Maximale massa", f"{max_massa} g")
    col3.metric("Totaal aantal metingen", totaal_metingen)


st.header("Frequentieverdeling")
st.markdown("De data wordt ingedeeld in klasse-intervallen voor statistische analyse.")

# Gebruiker kan het aantal klasse-intervallen aanpassen
st.sidebar.subheader("Frequentieverdeling instellingen")
custom_bins = st.sidebar.checkbox("Pas aantal klasse-intervallen handmatig aan")

if custom_bins:
    bins = st.sidebar.slider("Aantal klasse-intervallen", min_value=3, max_value=20, 
                           value=math.ceil(np.sqrt(totaal_metingen)), step=1)
else:
    bins = math.ceil(np.sqrt(totaal_metingen))

st.write(f'Aantal klasse-intervallen: {bins}')

# linspace genereert een array van waarden tussen min_massa en max_massa, inclusief de grenzen
klassengrenzen = np.linspace(min_massa, max_massa, bins + 1)
# pd.cut maakt klasse-intervallen op basis van de klassengrenzen
df['klasse'] = pd.cut(df['massa'], bins=klassengrenzen, include_lowest=True, right=True)
# Bereken de breedte van de klassen
klasse_breedtes = df['klasse'].apply(lambda x: x.right - x.left)
st.write("Breedte van de klassen (alleen unieke waarden):", klasse_breedtes.unique()[0], "gram")

# 1. Absolute frequentie - hoe vaak komt elke klasse voor in de dataset
absolute_frequentie = df['klasse'].value_counts().sort_index()
# 2. Relatieve frequentie - absolute frequentie gedeeld door het totaal aantal metingen
relatieve_frequentie = absolute_frequentie / totaal_metingen
# 3. Cumulatieve frequentie - som van de absolute frequenties tot en met de huidige klasse
cumulatieve_frequentie = absolute_frequentie.cumsum()

# Toon de frequentieverdelingen in een tabblad weergave
tab1, tab2, tab3 = st.tabs(["Absolute frequentie", "Relatieve frequentie", "Cumulatieve frequentie"])

with tab1:
    st.write("Aantal aardappels in elke massa-klasse:")
    st.dataframe(absolute_frequentie.reset_index().rename(columns={'index': 'Klasse', 'klasse': 'Absolute frequentie'}))
    
    # Plot van absolute frequentieverdeling
    fig, ax = plt.subplots(figsize=(10, 6))
    absolute_frequentie.plot(kind='bar', ax=ax)
    ax.set_title('Absolute frequentieverdeling')
    ax.set_xlabel('Massa klasse (g)')
    ax.set_ylabel('Aantal aardappels')
    st.pyplot(fig)

with tab2:
    st.write("Percentage aardappels in elke massa-klasse:")
    st.dataframe(relatieve_frequentie.reset_index().rename(columns={'index': 'Klasse', 'klasse': 'Relatieve frequentie'}))
    
    # Plot van relatieve frequentieverdeling
    fig, ax = plt.subplots(figsize=(10, 6))
    relatieve_frequentie.plot(kind='bar', ax=ax)
    ax.set_title('Relatieve frequentieverdeling')
    ax.set_xlabel('Massa klasse (g)')
    ax.set_ylabel('Percentage aardappels')
    # Percentages op de y-as
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    st.pyplot(fig)

with tab3:
    st.write("Cumulatieve verdeling van aardappel-massa's:")
    st.dataframe(cumulatieve_frequentie.reset_index().rename(columns={'index': 'Klasse', 'klasse': 'Cumulatieve frequentie'}))
    
    # Plot van cumulatieve frequentieverdeling
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulatieve_frequentie.plot(kind='bar', ax=ax)
    ax.set_title('Cumulatieve frequentieverdeling')
    ax.set_xlabel('Massa klasse (g)')
    ax.set_ylabel('Cumulatief aantal aardappels')
    st.pyplot(fig)

st.header("Statistische berekeningen")
st.subheader("Deel 1: Statistieken op basis van losse metingen")

# Bepaal het gemiddelde van de massa op basis van losse metingen
gemiddelde_massa_los = (df['massa'].sum() / totaal_metingen) if totaal_metingen > 0 else 0

with st.expander("Gemiddelde massa berekening", expanded=True):
    st.markdown("""
    ### Gemiddelde berekening
    Formule: $\\bar{x} = \\frac{\\sum_{i=1}^{n} x_i}{n}$
    
    Waarbij:
    - $x_i$ = individuele massa waarde
    - $n$ = totaal aantal metingen
    """)
    st.metric("Gemiddelde massa (losse metingen)", f"{gemiddelde_massa_los:.2f} g")

# Bepaal de steekproef standaardafwijking van de massa (losse metingen)
with st.expander("Standaardafwijking berekening", expanded=True):
    st.markdown("""
    ### Standaardafwijking berekening
    
    Formule steekproefvariantie: $s^2 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}{n-1}$
    
    Formule steekproefstandaardafwijking: $s = \\sqrt{s^2}$
    """)

    # Stap 1: Bereken de gekwadrateerde afwijkingen van de massa (losse metingen)
    gekwadrateerde_afwijkingen_los = []
    afwijkingen_los = []
    for xi in df['massa']:
        afwijking_los = xi - gemiddelde_massa_los
        afwijkingen_los.append(afwijking_los)
        gekwadrateerde_afwijkingen_los.append(afwijking_los ** 2)
    
    # Stap 2: Sommeer de gekwadrateerde afwijkingen
    som_gekwadrateerde_afwijkingen_los = sum(gekwadrateerde_afwijkingen_los)
    
    # Toon afwijkingen in een tabel als gewenst
    if st.checkbox("Toon afwijkingen van het gemiddelde"):
        afwijkingen_df = pd.DataFrame({
            'Massa (g)': df['massa'],
            'Afwijking van gemiddelde (g)': afwijkingen_los,
            'Gekwadrateerde afwijking': gekwadrateerde_afwijkingen_los
        })
        st.dataframe(afwijkingen_df)
    
    st.write(f"Som van gekwadrateerde afwijkingen: {som_gekwadrateerde_afwijkingen_los:.2f}")
    
    # Stap 3: Bereken de steekproefvariantie
    if totaal_metingen > 1:
        steekproef_variantie_los = som_gekwadrateerde_afwijkingen_los / (totaal_metingen - 1)
    else:
        steekproef_variantie_los = 0
    
    st.write(f"Steekproefvariantie: {steekproef_variantie_los:.2f}")
    
    # Stap 4: Bereken de steekproefstandaardafwijking
    if steekproef_variantie_los >= 0:
        handmatige_standaardafwijking_los = math.sqrt(steekproef_variantie_los)
    else:
        handmatige_standaardafwijking_los = 0
    
    st.metric("Steekproefstandaardafwijking (handmatig berekend)", f"{handmatige_standaardafwijking_los:.2f} g")
    
    # Ter controle met pandas .std() functie
    pandas_standaardafwijking_los = df['massa'].std(ddof=1)  # ddof=1 voor steekproef standaardafwijking
    st.metric("Steekproefstandaardafwijking (pandas)", f"{pandas_standaardafwijking_los:.2f} g")

st.subheader("Deel 2: Statistieken op basis van frequentieverdeling")

# Stap 1: Bepaal de klasse-middens
klasse_middens = []
for interval in absolute_frequentie.index:
    ondergrens = interval.left
    bovengrens = interval.right
    klasse_midden = (ondergrens + bovengrens) / 2
    klasse_middens.append(klasse_midden)

# Converteer naar numpy array voor verdere berekeningen
klasse_middens_np = np.array(klasse_middens)

with st.expander("Berekening van klasse-middens", expanded=True):
    st.markdown("""
    ### Klasse-middens
    Voor elke klasse berekenen we het midden door het gemiddelde van de ondergrens en bovengrens te nemen:
    
    $m_i = \\frac{\\text{ondergrens} + \\text{bovengrens}}{2}$
    """)
    
    # Weergave voor klasse-middens
    frequentietabel_middens = pd.DataFrame({
        'Klasse': absolute_frequentie.index,
        'Klassemidden (mᵢ)': klasse_middens_np
    })
    st.dataframe(frequentietabel_middens)
    
    st.write("Klassemiddens als lijst:", klasse_middens_np.tolist())

# Stap 2: Bepaal het gemiddelde op basis van de frequentieverdeling
with st.expander("Gemiddelde op basis van frequentieverdeling", expanded=True):
    st.markdown("""
    ### Gemiddelde berekening op basis van frequentieverdeling
    
    Formule: $\\bar{x}_{freq} \\approx \\frac{\\sum(f_i \\cdot m_i)}{n}$
    
    Waarbij:
    - $f_i$ = absolute frequentie van klasse i
    - $m_i$ = klassemidden van klasse i
    - $n$ = totaal aantal metingen
    """)
    
    # Bereken fᵢ * mᵢ voor elke klasse
    product_fi_mi = absolute_frequentie.values * klasse_middens_np
    
    # Presenteer data in een DataFrame
    tabel_fi_mi = pd.DataFrame({
        'Klasse': absolute_frequentie.index,
        'fᵢ (Absolute frequentie)': absolute_frequentie.values,
        'mᵢ (Klassemidden)': klasse_middens_np,
        'fᵢ * mᵢ': product_fi_mi
    })
    st.dataframe(tabel_fi_mi)
    
    # Sommeer de producten
    som_product_fi_mi = product_fi_mi.sum()
    st.write(f"Som van producten Σ(fᵢ * mᵢ): {som_product_fi_mi:.2f}")
    
    # Bereken het gemiddelde
    if totaal_metingen > 0:
        gemiddelde_massa_freq = som_product_fi_mi / totaal_metingen
        st.write(f"Totaal aantal metingen (n): {totaal_metingen}")
        st.write(f"Benaderd gemiddelde (x̄_freq) = {som_product_fi_mi:.4f}/{totaal_metingen} = {gemiddelde_massa_freq:.2f} g")
        
        # Vergelijking met het gemiddelde op basis van losse metingen
        verschil = abs(gemiddelde_massa_freq - gemiddelde_massa_los)
        st.info(f"Het verschil tussen het gemiddelde berekend op basis van losse metingen ({gemiddelde_massa_los:.2f} g) en op basis van frequentieverdeling ({gemiddelde_massa_freq:.2f} g) is {verschil:.2f} g.")
    else:
        gemiddelde_massa_freq = 0
        st.error("Totaal aantal metingen is 0, kan geen gemiddelde berekenen.")    
    
# Stap 3: Bereken de standaardafwijking op basis van frequentieverdeling
with st.expander("Standaardafwijking op basis van frequentieverdeling", expanded=True):
    st.markdown("""
    ### Standaardafwijkingsberekening op basis van frequentieverdeling
    
    Formule: $s_{freq} \\approx \\sqrt{\\frac{\\sum f_i \\cdot (m_i - \\bar{x}_{freq})^2}{n-1}}$
    
    Waarbij:
    - $f_i$ = absolute frequentie van klasse i
    - $m_i$ = klassemidden van klasse i
    - $\\bar{x}_{freq}$ = gemiddelde op basis van frequentieverdeling
    - $n$ = totaal aantal metingen
    """)
    
    # Controleer of het gemiddelde is berekend en n > 1 is
    if totaal_metingen > 1 and 'gemiddelde_massa_freq' in locals() and gemiddelde_massa_freq is not None:
        # Bereken de afwijking van elk klassemidden tot het gemiddelde
        afwijking_midden_vs_gemiddelde_freq = klasse_middens_np - gemiddelde_massa_freq
        
        # Kwadrateer de afwijkingen
        gekwadrateerde_afwijkingen_freq = afwijking_midden_vs_gemiddelde_freq ** 2
        
        # Vermenigvuldig met de frequentie van de klasse
        product_fi_gekwadrateerde_afwijkingen = absolute_frequentie.values * gekwadrateerde_afwijkingen_freq
        
        # Toon tussenstappen in een DataFrame
        tabel_variantie_stappen = pd.DataFrame({
            'Klasse': absolute_frequentie.index,
            'fᵢ (Absolute frequentie)': absolute_frequentie.values,
            'mᵢ (Klassemidden)': klasse_middens_np,
            'mᵢ - x̄_freq': afwijking_midden_vs_gemiddelde_freq,
            '(mᵢ - x̄_freq)²': gekwadrateerde_afwijkingen_freq,
            'fᵢ * (mᵢ - x̄_freq)²': product_fi_gekwadrateerde_afwijkingen
        })
        st.dataframe(tabel_variantie_stappen)
        
        # Sommeer de producten
        som_product_fi_gekwadrateerde_afwijkingen = product_fi_gekwadrateerde_afwijkingen.sum()
        st.write(f"Som van producten Σ(fᵢ * (mᵢ - x̄_freq)²): {som_product_fi_gekwadrateerde_afwijkingen:.2f}")
        
        # Bereken de steekproefvariantie
        steekproef_variantie_freq = som_product_fi_gekwadrateerde_afwijkingen / (totaal_metingen - 1)
        st.write(f"Steekproefvariantie: {steekproef_variantie_freq:.2f}")
        
        # Bereken de steekproefstandaardafwijking
        if steekproef_variantie_freq >= 0:
            handmatige_standaardafwijking_freq = math.sqrt(steekproef_variantie_freq)
            st.metric("Steekproefstandaardafwijking (frequentieverdeling)", f"{handmatige_standaardafwijking_freq:.2f} g")
            
            # Vergelijking met standaardafwijking op basis van losse metingen
            verschil_sd = abs(handmatige_standaardafwijking_freq - handmatige_standaardafwijking_los)
            st.info(f"Het verschil tussen de standaardafwijking berekend op basis van losse metingen ({handmatige_standaardafwijking_los:.2f} g) en op basis van frequentieverdeling ({handmatige_standaardafwijking_freq:.2f} g) is {verschil_sd:.2f} g.")
        else:
            st.error("Standaardafwijking kan niet worden berekend (negatieve variantie).")
    else:
        st.error("Standaardafwijking kan niet worden berekend (onvoldoende metingen of gemiddelde niet beschikbaar).")
        # Definieer de variabele om fouten in de code verderop te voorkomen
        handmatige_standaardafwijking_freq = 0

st.header("Normal Probability Plot")
st.markdown("""
Een Normal Probability Plot (ook bekend als Q-Q plot) helpt om te beoordelen of de data ongeveer normaal verdeeld zijn. 
Als de punten dicht bij de regressielijn liggen, suggereert dit dat de data een normale verdeling volgen.
""")

# Maak tabs voor de plot en de data
tab1, tab2 = st.tabs(["Plot", "Data"])

with tab1:
    # Bereken de data voor de pplot
    (osm, osr), (slope, intercept, r_value) = stats.probplot(df['massa'], dist="norm", plot=None)
    
    # Maak de figuur
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Maak de scatter plot
    ax.scatter(osr, osm, label='Geobserveerde waarden', s=50)
    
    # Voeg de regressielijn toe
    x_line_plot = np.linspace(osr.min(), osr.max(), 100)
    
    if abs(float(slope)) > 1e-9:  # Voorkom deling door nul
        y_line_plot = (x_line_plot - intercept) / slope
        ax.plot(x_line_plot, y_line_plot, 'r', label='Regressielijn (kleinste kwadraten)', linewidth=2)
    else:
        st.warning("Helling van de regressielijn is zeer klein, de lijn kan mogelijk niet correct worden weergegeven.")
    
    ax.set_title('Normal Probability Plot van Aardappel Massa (assen gedraaid)')
    ax.set_xlabel('Theoretische Waarden (z-scores)')
    ax.set_ylabel('Geobserveerde Waarden (Massa g)')
    ax.grid(True)
    
    # Toon R-kwadraat op de plot
    r_squared = float(r_value)**2
    ax.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    ax.legend()
    st.pyplot(fig)
    
    # Interpretatie
    if r_squared > 0.95:
        st.success(f"R² = {r_squared:.4f}: De data volgt zeer waarschijnlijk een normale verdeling.")
    elif r_squared > 0.90:
        st.info(f"R² = {r_squared:.4f}: De data volgt redelijk goed een normale verdeling.")
    else:
        st.warning(f"R² = {r_squared:.4f}: De data wijkt mogelijk af van een normale verdeling.")

with tab2:
    # Toon de tabel met corresponderende data
    data_tabel = pd.DataFrame({
        'Geobserveerde Waarde (Massa g)': osm,
        'Theoretische Waarde (z-score)': osr
    }).round(4)
    
    st.dataframe(data_tabel)
    
    # Toon de regressieparameters
    st.subheader("Parameters van de regressielijn")
    st.markdown(f"""
    Formule: Geobserveerde waarde = slope × Theoretische waarde + intercept
    
    - Helling (slope): {float(slope):.4f}
    - Onderschepping (intercept): {float(intercept):.4f}
    - Correlatiecoëfficiënt (r-value): {float(r_value):.4f}
    - R-kwadraat (R²): {r_squared:.4f}
    """)
    
    # Optie om de plot-data te downloaden
    csv = data_tabel.to_csv(index=False)
    st.download_button(
        label="Download plot data als CSV",
        data=csv,
        file_name="normal_probability_plot_data.csv",
        mime="text/csv"
    )

st.header("Samenvatting Bevindingen")

# Samenvatting van de bevindingen in een mooi formaat
st.markdown(f"""
### Kerncijfers
- Gemiddelde massa op basis van losse metingen: **{gemiddelde_massa_los:.2f} g**
- Gemiddelde massa op basis van frequentieverdeling: **{gemiddelde_massa_freq:.2f} g**
- Standaardafwijking op basis van losse metingen: **{handmatige_standaardafwijking_los:.2f} g**
- Standaardafwijking op basis van frequentieverdeling: **{handmatige_standaardafwijking_freq:.2f} g**
""")

# Betrouwbaarheidsintervallen sectie
st.header("Betrouwbaarheidsinterval voor het Gemiddelde")
st.markdown("""
Een betrouwbaarheidsinterval geeft een bereik waarin de werkelijke populatie-parameter met een bepaalde waarschijnlijkheid ligt.
""")

# Gebruiker kan het betrouwbaarheidsniveau kiezen
confidence_level = st.slider(
    "Kies een betrouwbaarheidsniveau",
    min_value=80,
    max_value=99,
    value=95,
    step=1,
    format="%d%%"
)

# Bereken het betrouwbaarheidsinterval voor het gemiddelde
alpha = (100 - confidence_level) / 100  # bijv. 95% -> alpha = 0.05
n = totaal_metingen
df_t = n - 1  # vrijheidsgraden voor t-verdeling

# Verkrijg de kritieke t-waarde
t_kritiek = stats.t.ppf(1 - alpha / 2, df_t)

# Maak twee kolommen voor de gegevens en het resultaat
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Gegevens")
    st.write(f"Steekproefgemiddelde (x̄): {gemiddelde_massa_los:.2f} g")
    st.write(f"Steekproefstandaardafwijking (s): {handmatige_standaardafwijking_los:.2f} g")
    st.write(f"Steekproefgrootte (n): {n}")
    st.write(f"Vrijheidsgraden (n-1): {df_t}")
    st.write(f"Kritieke t-waarde (t_(α/2, n-1)): {t_kritiek:.4f}")

# Standaardfout van het gemiddelde
standaardfout = handmatige_standaardafwijking_los / math.sqrt(n)

# Bereken de foutmarge
foutmarge = t_kritiek * standaardfout

# Bereken het betrouwbaarheidsinterval
ondergrens = float(gemiddelde_massa_los) - float(foutmarge)
bovengrens = float(gemiddelde_massa_los) + float(foutmarge)

with col2:
    st.markdown("### Berekening")
    st.latex(r'\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}')
    st.write(f"Standaardfout (s/√n): {standaardfout:.4f}")
    st.write(f"Foutmarge: {foutmarge:.4f}")
    st.success(f"{confidence_level}% Betrouwbaarheidsinterval: [{ondergrens:.2f}, {bovengrens:.2f}] g")
    
    # Visualisatie met een gauge chart
    fig_gauge = {
        "data": [
            {
                "type": "indicator",
                "mode": "gauge+number",
                "value": gemiddelde_massa_los,
                "title": {"text": f"Gemiddelde met {confidence_level}% BI"},
                "gauge": {
                    "axis": {"range": [None, max_massa * 1.1], "tickwidth": 1},
                    "bar": {"color": "darkblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [min_massa, ondergrens], "color": "lightgray"},
                        {"range": [ondergrens, bovengrens], "color": "lightblue"},
                        {"range": [bovengrens, max_massa], "color": "lightgray"}
                    ],
                }
            }
        ],
        "layout": {"height": 300, "margin": {"t": 30, "b": 0, "l": 30, "r": 30}}
    }
    st.plotly_chart(fig_gauge, use_container_width=True)

# Conclusies
st.subheader("Conclusies")
st.markdown(f"""
1. De steekproef van **{n} aardappelen** heeft een gemiddelde massa van **{gemiddelde_massa_los:.2f} g**.
2. Met **{confidence_level}% betrouwbaarheid** kunnen we stellen dat het populatiegemiddelde van de aardappelmassa's ligt tussen **{ondergrens:.2f} g** en **{bovengrens:.2f} g**.
3. De standaardafwijking bedraagt **{handmatige_standaardafwijking_los:.2f} g**, wat duidt op de spreiding in de massa van individuele aardappelen.
""")

st.header("Betrouwbaarheidsinterval voor de Standaarddeviatie")
st.markdown("""
Het betrouwbaarheidsinterval voor de standaarddeviatie geeft aan binnen welke grenzen de werkelijke populatie-standaardafwijking (σ) waarschijnlijk ligt.
""")

# Als de gebruiker het betrouwbaarheidsniveau al gekozen heeft, gebruik dat, anders standaard 95%
if 'confidence_level' not in locals():
    confidence_level = st.slider(
        "Kies een betrouwbaarheidsniveau",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        key="conf_level_sd",
        format="%d%%"
    )

# Bereken het betrouwbaarheidsinterval voor de standaarddeviatie
alpha = (100 - confidence_level) / 100  # bijv. 95% -> alpha = 0.05
n = totaal_metingen
df_chi2 = n - 1  # vrijheidsgraden voor chi-kwadraat verdeling

# Kritieke waarden van chi-kwadraat
chi2_lower = stats.chi2.ppf(alpha/2, df_chi2)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df_chi2)

# Bereken de grenzen van het betrouwbaarheidsinterval
sd_lower = math.sqrt((df_chi2 * steekproef_variantie_los) / chi2_upper)
sd_upper = math.sqrt((df_chi2 * steekproef_variantie_los) / chi2_lower)

# Maak twee kolommen voor de gegevens en het resultaat
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Gegevens")
    st.write(f"Steekproefstandaardafwijking (s): {handmatige_standaardafwijking_los:.4f} g")
    st.write(f"Steekproefvariantie (s²): {steekproef_variantie_los:.4f}")
    st.write(f"Vrijheidsgraden (n-1): {df_chi2}")
    st.write(f"Chi-kwadraat ondergrens: {chi2_lower:.4f}")
    st.write(f"Chi-kwadraat bovengrens: {chi2_upper:.4f}")

with col2:
    st.markdown("### Berekening")
    st.latex(r'\sqrt{\frac{(n-1)s^2}{\chi^2_{\alpha/2,n-1}}} < \sigma < \sqrt{\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,n-1}}}')
    st.write(f"Ondergrens: √[({df_chi2} × {steekproef_variantie_los:.4f}) / {chi2_upper:.4f}] = {sd_lower:.4f}")
    st.write(f"Bovengrens: √[({df_chi2} × {steekproef_variantie_los:.4f}) / {chi2_lower:.4f}] = {sd_upper:.4f}")
    st.success(f"{confidence_level}% Betrouwbaarheidsinterval voor σ: [{sd_lower:.2f}, {sd_upper:.2f}] g")

st.info(f"""
De populatie standaardafwijking (σ) van de aardappelmassa's ligt met {confidence_level}% zekerheid tussen {sd_lower:.2f} g en {sd_upper:.2f} g.
Dit betekent dat de werkelijke variabiliteit in aardappelmassa's kan afwijken van de steekproefschatting ({handmatige_standaardafwijking_los:.2f} g), maar waarschijnlijk binnen deze grenzen valt.
""")

# Kans op afwijking van het gemiddelde sectie
st.header("Kans op Afwijking van het Gemiddelde")
st.markdown("""
Als de data normaal verdeeld is, kunnen we de kans berekenen dat een willekeurige aardappel meer dan een bepaald aantal
standaarddeviaties van het gemiddelde afwijkt.
""")

# Interactief element: laat gebruiker kiezen hoeveel standaarddeviaties
z_score = st.slider(
    "Aantal standaarddeviaties van het gemiddelde",
    min_value=0.5,
    max_value=3.0,
    value=1.8,
    step=0.1
)

# Grenswaarden berekenen
ondergrens_z = float(gemiddelde_massa_los) - z_score * float(handmatige_standaardafwijking_los)
bovengrens_z = float(gemiddelde_massa_los) + z_score * float(handmatige_standaardafwijking_los)

# Kansen berekenen
kans_onder = stats.norm.cdf(-z_score)
kans_boven = 1 - stats.norm.cdf(z_score)
kans_totaal = kans_onder + kans_boven

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Grenswaarden")
    st.write(f"Gemiddelde (µ): {gemiddelde_massa_los:.2f} g")
    st.write(f"Standaardafwijking (σ): {handmatige_standaardafwijking_los:.2f} g")
    st.write(f"Ondergrens (µ - {z_score}σ): {ondergrens_z:.2f} g")
    st.write(f"Bovengrens (µ + {z_score}σ): {bovengrens_z:.2f} g")

with col2:
    st.markdown("### Kansen")
    st.write(f"Kans dat een aardappel lichter is dan {ondergrens_z:.2f} g: {kans_onder:.4f} = {kans_onder*100:.2f}%")
    st.write(f"Kans dat een aardappel zwaarder is dan {bovengrens_z:.2f} g: {kans_boven:.4f} = {kans_boven*100:.2f}%")
    st.write(f"Totale kans op afwijking van meer dan {z_score} standaarddeviaties: {kans_totaal:.4f} = {kans_totaal*100:.2f}%")

# Visualisatie van de normale verdeling
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(float(gemiddelde_massa_los) - 4*float(handmatige_standaardafwijking_los), 
                float(gemiddelde_massa_los) + 4*float(handmatige_standaardafwijking_los), 1000)
y = stats.norm.pdf(x, float(gemiddelde_massa_los), float(handmatige_standaardafwijking_los))

# Plot de normale verdeling
ax.plot(x, y, 'b-', label='Normale verdeling')

# Kleur de gebieden links van de ondergrens
x_lower = np.linspace(min(x), float(ondergrens_z), 100)
y_lower = stats.norm.pdf(x_lower, float(gemiddelde_massa_los), float(handmatige_standaardafwijking_los))
ax.fill_between(x_lower, y_lower, color='red', alpha=0.3, label=f'Kans op afwijking < -{z_score}σ')

# Kleur de gebieden rechts van de bovengrens
x_upper = np.linspace(float(bovengrens_z), max(x), 100)
y_upper = stats.norm.pdf(x_upper, float(gemiddelde_massa_los), float(handmatige_standaardafwijking_los))
ax.fill_between(x_upper, y_upper, color='red', alpha=0.3, label=f'Kans op afwijking > {z_score}σ')

# Voeg verticale lijnen toe voor het gemiddelde en de grenzen
ax.axvline(x=gemiddelde_massa_los, color='green', linestyle='-', label='Gemiddelde')
ax.axvline(x=ondergrens_z, color='red', linestyle='--', label=f'µ - {z_score}σ')
ax.axvline(x=bovengrens_z, color='red', linestyle='--', label=f'µ + {z_score}σ')

ax.set_title(f'Normale verdeling van aardappelmassa met afwijking > {z_score}σ')
ax.set_xlabel('Massa (g)')
ax.set_ylabel('Waarschijnlijkheidsdichtheid')
ax.legend()

st.pyplot(fig)

st.success(f"""
De kans dat een willekeurige aardappel meer dan {z_score} standaarddeviaties van het gemiddelde afwijkt is {kans_totaal*100:.2f}%.
Dit betekent dat ongeveer {kans_totaal*100:.2f}% van alle aardappelen een massa heeft die kleiner is dan {ondergrens_z:.2f} g of groter dan {bovengrens_z:.2f} g.
""")

st.header("Conclusies en Aanbevelingen")
st.markdown("""
Op basis van de statistische analyse van de aardappelmassa's kunnen we de volgende conclusies trekken:

1. **Centrale tendens**: De gemiddelde massa van de aardappels is ongeveer geïdentificeerd en ligt binnen een betrouwbaarheidsinterval.
2. **Variabiliteit**: De standaardafwijking geeft aan hoe sterk de massa's variëren rond het gemiddelde.
3. **Normaliteit**: De Normal Probability Plot laat zien in hoeverre de data een normale verdeling volgt.
4. **Kans op uitschieters**: We hebben de kans berekend dat een willekeurige aardappel sterk afwijkt van het gemiddelde.

### Aanbevelingen
- De statistische kennis uit deze app kan worden toegepast op andere datasets.
- Voor toekomstige analyses kunnen meer geavanceerde statistische methoden worden overwogen.
""")

# Download link voor het rapport
report = io.StringIO()
report.write(f"""# Statistisch Rapport Aardappelmassa's

## Basisinformatie
- Aantal metingen: {totaal_metingen}
- Minimale massa: {min_massa} g
- Maximale massa: {max_massa} g

## Centrale tendens
- Gemiddelde (losse metingen): {gemiddelde_massa_los:.2f} g
- Gemiddelde (frequentieverdeling): {gemiddelde_massa_freq:.2f} g

## Spreiding
- Standaardafwijking (losse metingen): {handmatige_standaardafwijking_los:.2f} g
- Standaardafwijking (frequentieverdeling): {getattr(locals().get('handmatige_standaardafwijking_freq', 0), '__float__', lambda: 0)():.2f} g

## Betrouwbaarheidsintervallen
- {confidence_level}% BI voor het gemiddelde: [{ondergrens:.2f}, {bovengrens:.2f}] g
- {confidence_level}% BI voor de standaardafwijking: [{sd_lower:.2f}, {sd_upper:.2f}] g

## Kans op afwijking van het gemiddelde
- Kans op afwijking > {z_score} standaarddeviaties: {kans_totaal*100:.2f}%

Dit rapport is gegenereerd op {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
""")

st.download_button(
    label="Download rapport als tekstbestand",
    data=report.getvalue(),
    file_name="aardappel_statistiek_rapport.txt",
    mime="text/plain"
)

# Sidebar met contactgegevens en extra info
st.sidebar.title("Over deze app")
st.sidebar.info("""
Deze app is ontwikkeld als een educatief hulpmiddel voor het begrijpen van basis statistiek.

**Kenmerken:**
- Interactieve visualisaties
- Stapsgewijze berekeningen
- Downloadbaar rapport

Voor vragen of suggesties, neem contact op met de ontwikkelaar.
""")

# Footer
st.markdown("---")
st.markdown("© 2025 | Ontwikkeld met Streamlit")
