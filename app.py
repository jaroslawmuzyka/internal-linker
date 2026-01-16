import streamlit as st
import pandas as pd
import numpy as np
import io
from collections import defaultdict

# Konfiguracja strony
st.set_page_config(page_title="AI Internal Linker", page_icon="🔗", layout="wide")

# --- Funkcje pomocnicze ---

def parse_embedding(emb_str):
    """Konwertuje ciąg tekstowy '0.123, 0.456...' na wektor numpy."""
    try:
        if isinstance(emb_str, str):
            # Usuwamy ewentualne nawiasy klamrowe czy spacje
            clean_str = emb_str.replace('[', '').replace(']', '').strip()
            return np.fromstring(clean_str, sep=',')
        return np.array([])
    except:
        return np.array([])

def cosine_similarity_matrix(source_vecs, target_vecs):
    """
    Oblicza podobieństwo cosinusowe między dwiema macierzami wektorów.
    Zwraca macierz o wymiarach (len(source), len(target)).
    """
    # Normalizacja wektorów (L2 norm)
    source_norm = np.linalg.norm(source_vecs, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target_vecs, axis=1, keepdims=True)
    
    # Unikanie dzielenia przez zero
    source_norm[source_norm == 0] = 1
    target_norm[target_norm == 0] = 1
    
    # Obliczenie cosinusa: (A . B) / (|A| * |B|)
    similarity = np.dot(source_vecs, target_vecs.T) / np.dot(source_norm, target_norm.T)
    return similarity

def load_data(uploaded_file):
    """Wczytuje plik segmentu i sprawdza kolumny."""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Mapowanie kolumn (elastyczne podejście do nazw)
        required_cols = {
            'address': ['Address', 'URL', 'Adres'],
            'title': ['Title 1', 'Title', 'Tytuł'],
            'h1': ['H1-1', 'H1', 'Nagłówek 1'],
            'emb': ['Extract embeddings', 'Embedding', 'Vector']
        }
        
        col_map = {}
        for key, possible_names in required_cols.items():
            found = False
            for name in possible_names:
                # Szukamy kolumny zawierającej daną nazwę (case insensitive)
                match = next((c for c in df.columns if name.lower() in c.lower()), None)
                if match:
                    col_map[key] = match
                    found = True
                    break
            if not found:
                return None, f"Brak wymaganej kolumny dla: {key} (szukano: {possible_names})"

        # Filtrowanie i parsowanie
        df = df.dropna(subset=[col_map['address'], col_map['emb']])
        
        # Parsowanie embeddingów do nowej kolumny
        df['parsed_emb'] = df[col_map['emb']].apply(parse_embedding)
        
        # Usuwanie błędnych embeddingów (pustych)
        df = df[df['parsed_emb'].apply(lambda x: x.size > 0)]
        
        # Ustandaryzowanie nazw kolumn do dalszej pracy
        clean_df = pd.DataFrame({
            'Address': df[col_map['address']].astype(str).str.strip(),
            'Title': df[col_map['title']].fillna('').astype(str),
            'H1': df[col_map['h1']].fillna('').astype(str),
            'Embedding': df['parsed_emb']
        })
        
        return clean_df, None
        
    except Exception as e:
        return None, str(e)

def load_anchors(uploaded_file):
    """Wczytuje plik anchorów i zwraca słownik {url: [anchor1, anchor2]}."""
    try:
        df = pd.read_excel(uploaded_file)
        # Szukanie kolumn
        url_col = next((c for c in df.columns if 'url' in c.lower()), None)
        anc_col = next((c for c in df.columns if 'anchor' in c.lower()), None)
        
        if not url_col or not anc_col:
            return None, "Nie znaleziono kolumn 'URL' lub 'anchor' w pliku anchorów."
            
        anchors_map = defaultdict(list)
        for _, row in df.iterrows():
            u = str(row[url_col]).strip()
            a = str(row[anc_col]).strip()
            if u and a:
                anchors_map[u].append(a)
        
        return anchors_map, None
    except Exception as e:
        return None, str(e)

# --- Interfejs Użytkownika ---

st.title("🔗 AI Internal Linking Strategy")
st.markdown("""
To narzędzie generuje strategię linkowania wewnętrznego na podstawie **podobieństwa semantycznego (embeddingów)**.
Wgraj pliki z segmentami (np. Kategorie, Blog), a AI dobierze najbardziej pasujące podstrony.
""")

with st.expander("ℹ️ Instrukcja i format plików"):
    st.markdown("""
    1. **Pliki segmentów (.xlsx):** Muszą zawierać kolumny: `Address`, `Title 1`, `H1-1`, `Extract embeddings`.
    2. **Plik anchorów (opcjonalny .xlsx):** Kolumny: `URL`, `anchor`.
    3. **Działanie:** Wybierz segment główny (źródło linków) i segmenty docelowe. System znajdzie najlepsze dopasowania.
    """)

# --- Krok 1: Konfiguracja ---

col1, col2 = st.columns(2)
with col1:
    num_segments = st.number_input("Liczba segmentów (grup stron)", min_value=2, max_value=10, value=2)
with col2:
    limit_suggestions = st.number_input("Liczba linków na artykuł", min_value=1, max_value=20, value=3)

# Przechowywanie wgranych plików w sesji (aby nie znikały przy przeładowaniu)
if 'segment_files' not in st.session_state:
    st.session_state['segment_files'] = {}

st.subheader("📂 Wgraj pliki segmentów")

segments_data = [] # Lista słowników: {'name': str, 'df': DataFrame}
has_errors = False

for i in range(num_segments):
    c1, c2 = st.columns([1, 2])
    with c1:
        seg_name = st.text_input(f"Nazwa segmentu {i+1}", value=f"Segment {i+1}", key=f"name_{i}")
    with c2:
        seg_file = st.file_uploader(f"Plik dla: {seg_name}", type=['xlsx'], key=f"file_{i}")
    
    if seg_file:
        df, error = load_data(seg_file)
        if error:
            st.error(f"Błąd w pliku '{seg_name}': {error}")
            has_errors = True
        else:
            segments_data.append({'name': seg_name, 'df': df})
            st.success(f"✅ Wczytano {len(df)} adresów.")

st.subheader("⚓ Anchory (Opcjonalne)")
anchor_file = st.file_uploader("Plik z anchorami (.xlsx)", type=['xlsx'])
anchors_map = {}
if anchor_file:
    a_map, a_err = load_anchors(anchor_file)
    if a_err:
        st.error(f"Błąd anchorów: {a_err}")
    else:
        anchors_map = a_map
        st.success(f"✅ Wczytano anchory dla {len(anchors_map)} adresów URL.")

# --- Krok 2: Uruchomienie ---

if len(segments_data) == num_segments and not has_errors:
    st.divider()
    
    # Wybór segmentu głównego
    segment_names = [s['name'] for s in segments_data]
    main_seg_idx = st.selectbox("Wybierz Segment Główny (skąd linkujemy?):", range(len(segment_names)), format_func=lambda x: segment_names[x])
    
    if st.button("🚀 Generuj Strategię Linkowania", type="primary"):
        with st.spinner("Analizuję wektory i obliczam podobieństwo..."):
            
            main_segment = segments_data[main_seg_idx]
            other_segments = [s for i, s in enumerate(segments_data) if i != main_seg_idx]
            
            # Przygotowanie macierzy wektorów dla segmentu głównego
            # Stackujemy wektory do macierzy numpy (N x Dymensje)
            main_vecs = np.vstack(main_segment['df']['Embedding'].values)
            
            results = []
            usage_counts = defaultdict(int) # Do rotacji anchorów: {url_target: count}
            
            # Zbieramy wszystkie URL z "innych" segmentów, aby znaleźć nielinkowane
            all_target_urls = set()
            linked_target_urls = set()
            
            # Główna pętla przetwarzania
            # Iterujemy po innych segmentach
            for target_seg in other_segments:
                target_df = target_seg['df']
                target_vecs = np.vstack(target_df['Embedding'].values)
                
                # Dodajemy do puli wszystkich URLi
                all_target_urls.update(target_df['Address'].tolist())
                
                # Obliczamy podobieństwo WSZYSTKO vs WSZYSTKO dla tej pary segmentów
                # Wynik to macierz: wiersze = main_urls, kolumny = target_urls
                sim_matrix = cosine_similarity_matrix(main_vecs, target_vecs)
                
                # Dla każdego adresu z Main Segment
                for idx, source_row in main_segment['df'].iterrows():
                    # Pobieramy wiersz podobieństw dla tego adresu
                    # idx może nie odpowiadać indeksowi macierzy jeśli df ma luki w indexie,
                    # więc bezpieczniej użyć iloc/reset_index, ale tutaj iterujemy po kolei
                    # Użyjmy licznika pętli
                    pass 

                # Podejście ziterowane po macierzy jest szybsze
                # sim_matrix[i] to podobieństwa dla i-tego adresu z main_segment
                
                for i in range(len(main_segment['df'])):
                    source_url = main_segment['df'].iloc[i]['Address']
                    
                    # Sortujemy wyniki dla tego wiersza (malejąco)
                    # argsort zwraca indeksy posortowane rosnąco, więc bierzemy od tyłu
                    scores = sim_matrix[i]
                    best_indices = np.argsort(scores)[::-1]
                    
                    # Bierzemy top N, pomijając ten sam URL (autolinkowanie)
                    count = 0
                    for target_idx in best_indices:
                        if count >= limit_suggestions:
                            break
                        
                        target_row = target_df.iloc[target_idx]
                        target_url = target_row['Address']
                        score = scores[target_idx]
                        
                        # Pomijamy autolinkowanie
                        if source_url == target_url:
                            continue
                            
                        # --- Logika Anchorów ---
                        # Sprawdzamy czy mamy zdefiniowane anchory dla TARGET URL
                        target_anchors = anchors_map.get(target_url, [])
                        
                        if target_anchors:
                            # Rotacja anchorów
                            anchor_idx = usage_counts[target_url] % len(target_anchors)
                            chosen_anchor = target_anchors[anchor_idx]
                        else:
                            chosen_anchor = "BRAK ANCHORA (Użyj tytułu)" # lub puste
                        
                        # Zwiększamy licznik użycia targetu
                        usage_counts[target_url] += 1
                        linked_target_urls.add(target_url)
                        
                        results.append({
                            'URL Źródłowy': source_url,
                            'Linkuje do': target_url,
                            'Segment Docelowy': target_seg['name'],
                            'Score': round(score, 4),
                            'Anchor': chosen_anchor,
                            'Tytuł Docelowy': target_row['Title']
                        })
                        count += 1

            # --- Generowanie wyników ---
            results_df = pd.DataFrame(results)
            
            # --- Generowanie nielinkowanych ---
            unlinked_list = list(all_target_urls - linked_target_urls)
            unlinked_df = pd.DataFrame(unlinked_list, columns=['Nielinkowany URL'])
            
            # Wyświetlanie
            st.success("✅ Analiza zakończona!")
            
            tab1, tab2 = st.tabs(["📊 Propozycje Linkowania", "🚫 Nielinkowane Adresy"])
            
            with tab1:
                st.dataframe(results_df, use_container_width=True)
                
                # Eksport
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Linki')
                
                st.download_button(
                    label="📥 Pobierz strategię (.xlsx)",
                    data=buffer.getvalue(),
                    file_name="strategia_linkowania.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            with tab2:
                st.write(f"Znaleziono {len(unlinked_df)} adresów, które nie otrzymały żadnego linku.")
                st.dataframe(unlinked_df, use_container_width=True)
                
                buffer_un = io.BytesIO()
                with pd.ExcelWriter(buffer_un, engine='xlsxwriter') as writer:
                    unlinked_df.to_excel(writer, index=False, sheet_name='Nielinkowane')
                    
                st.download_button(
                    label="📥 Pobierz nielinkowane (.xlsx)",
                    data=buffer_un.getvalue(),
                    file_name="nielinkowane_urls.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

elif len(segments_data) < num_segments:
    st.info("Wgraj wszystkie wymagane pliki segmentów, aby rozpocząć.")
