# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:40:30 2024

@author: jperezr
"""

import streamlit as st
import rispy
import bibtexparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</sytle>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Función para filtrar registros que contienen Título, Resumen y Fecha
def filter_records(records):
    valid_records = []
    invalid_records = []
    
    for record in records:
        title = record.get('title', None)
        abstract = record.get('abstract', None)
        year = record.get('year', None)
        
        if title and abstract and year:
            valid_records.append({'Título': title, 'Resumen': abstract, 'Fecha': int(year)})
        else:
            invalid_records.append(record)
    
    return valid_records, invalid_records

# Función para leer archivos .ris y extraer campos válidos
def parse_ris(files):
    valid_records = []
    invalid_records = []
    
    for file in files:
        with file:
            ris_data = file.read().decode('utf-8')
            try:
                parsed_records = rispy.loads(ris_data)
                valid, invalid = filter_records(parsed_records)
                valid_records.extend(valid)
                invalid_records.extend(invalid)
            except Exception as e:
                st.error(f"Error al procesar el archivo .ris: {e}")
    
    return pd.DataFrame(valid_records), len(invalid_records)

# Función para leer archivos .bib y extraer campos válidos
def parse_bib(files):
    valid_records = []
    invalid_records = []
    
    for file in files:
        with file:
            bib_data = file.read().decode('utf-8')
            try:
                bib_database = bibtexparser.loads(bib_data)
                for entry in bib_database.entries:
                    title = entry.get('title', None)
                    abstract = entry.get('abstract', None)
                    year = entry.get('year', None)
                    
                    if title and abstract and year:
                        valid_records.append({'Título': title, 'Resumen': abstract, 'Fecha': int(year)})
                    else:
                        invalid_records.append(entry)
            except Exception as e:
                st.error(f"Error al procesar el archivo .bib: {e}")
    
    return pd.DataFrame(valid_records), len(invalid_records)

# Función para buscar duplicados
def find_duplicates(df):
    duplicates = df[df.duplicated(subset=['Título', 'Resumen'], keep=False)]
    return duplicates

# Función para calcular la similitud entre Título y Resumen
def check_similarity(df, threshold=0.2):
    vectorizer = TfidfVectorizer(stop_words='english')
    similarities = []
    
    for index, row in df.iterrows():
        texts = [row['Título'], row['Resumen']]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarities.append(similarity)
    
    df['Similitud'] = similarities
    incoherent_records = df[df['Similitud'] < threshold]
    coherent_records = df[df['Similitud'] >= threshold]
    
    return incoherent_records, coherent_records

# Función para aplicar criterios de selección
def apply_selection_criteria(df):
    keyword_filter = df[df['Resumen'].str.contains('portfolio', case=False) | df['Resumen'].str.contains('optimization', case=False)]
    date_filter = keyword_filter[(keyword_filter['Fecha'] >= 2017) & (keyword_filter['Fecha'] <= 2024)]
    return date_filter

# Interfaz en Streamlit
st.title("Revisión Sistemática en Optimización de Portafolios utilizando Inteligencia Artificial")
st.subheader("Cargar y Analizar Archivos .ris y .bib")

uploaded_files = st.file_uploader("Cargar archivos .ris o .bib", accept_multiple_files=True, type=["ris", "bib"])

# Botón de descarga del PDF
pdf_file_path = "Reporte.pdf"  # Ruta del archivo PDF
with open(pdf_file_path, "rb") as pdf_file:
    st.sidebar.download_button(
        label="Descargar Reporte PDF",
        data=pdf_file,
        file_name="Reporte.pdf",
        mime="application/pdf"
    )


# Botón de descarga del PDF
pdf_file_path = "Reporte.pdf"  # Ruta del archivo PDF
with open(pdf_file_path, "rb") as pdf_file:
    st.sidebar.download_button(
        label="Descargar Reporte PDF",
        data=pdf_file,
        file_name="sysrev1_tex.pdf",
        mime="application/pdf"
    )


if uploaded_files:
    valid_ris_df = pd.DataFrame()
    valid_bib_df = pd.DataFrame()
    total_invalid = 0
    
    for file in uploaded_files:
        file_name = file.name
        if file_name.endswith('.ris'):
            ris_df, invalid_ris_count = parse_ris([file])
            valid_ris_df = pd.concat([valid_ris_df, ris_df])
            total_invalid += invalid_ris_count
        elif file_name.endswith('.bib'):
            bib_df, invalid_bib_count = parse_bib([file])
            valid_bib_df = pd.concat([valid_bib_df, bib_df])
            total_invalid += invalid_bib_count
    
    combined_df = pd.concat([valid_ris_df, valid_bib_df]).reset_index(drop=True)
    total_records = combined_df.shape[0] + total_invalid
    
    st.write(f"Total de registros subidos: {total_records}")
    st.write(f"Registros sin Título, Resumen o Fecha: {total_invalid}")
    
    valid_count = combined_df.shape[0]
    st.write(f"Registros con Título, Resumen y Fecha: {valid_count}")

    duplicates_df = find_duplicates(combined_df)
    st.write("Buscando duplicados por Título y Resumen...")
    st.write(f"Registros duplicados encontrados: {duplicates_df.shape[0]}")
    
    if not duplicates_df.empty:
        st.write("Se encontraron los siguientes duplicados:")
        st.dataframe(duplicates_df)
        
        combined_df = combined_df.drop_duplicates(subset=['Título', 'Resumen'], keep='first')
        
    st.write(f"Registros con Título, Resumen y Fecha después de eliminar duplicados: {combined_df.shape[0]}")
    
    if not combined_df.empty:
        st.write("Archivos procesados (con Título, Resumen y Fecha):")
        st.dataframe(combined_df)
        
        st.write("Verificando coherencia entre Título y Resumen...")
        incoherent_df, coherent_df = check_similarity(combined_df, threshold=0.2)
        
        st.write(f"Registros con baja coherencia entre Título y Resumen: {incoherent_df.shape[0]}")
        
        if not incoherent_df.empty:
            st.write("Los siguientes registros tienen baja coherencia entre Título y Resumen:")
            st.dataframe(incoherent_df)
        
        st.write(f"Registros con alta coherencia entre Título y Resumen: {coherent_df.shape[0]}")
        
        if not coherent_df.empty:
            st.write("Los siguientes registros tienen alta coherencia entre Título y Resumen:")
            st.dataframe(coherent_df)
            
            st.write("Criterios de selección utilizados:")
            st.write("- Los resúmenes deben contener las palabras clave 'portfolio' u 'optimization'.")
            st.write("- La fecha debe estar entre 2017 y 2024.")
            
            selected_df = apply_selection_criteria(coherent_df)
            st.write(f"Registros que cumplen con los criterios: {selected_df.shape[0]}")
            
            if not selected_df.empty:
                st.write("Los siguientes registros cumplen con los criterios establecidos:")
                st.dataframe(selected_df)

                # Gráfico de barras dinámico
                st.write("### Gráfico de barras de registros por año")
                fig = px.histogram(selected_df, x='Fecha', title='Número de Registros por Año', 
                                   labels={'Fecha': 'Año', 'count': 'Número de Registros'},
                                   color_discrete_sequence=['lightblue'])  # Define bar color

                fig.update_traces(marker=dict(line=dict(color='black', width=1.5)))  # Add border color and width

                st.plotly_chart(fig)

                # Agregar un selectbox para elegir la fecha
                unique_years = sorted(selected_df['Fecha'].unique())
                selected_year = st.selectbox("Selecciona un año para filtrar:", unique_years)

                # Filtrar el DataFrame por año seleccionado
                filtered_df = selected_df[selected_df['Fecha'] == selected_year]

                st.write(f"Registros para el año {selected_year}:")
                st.dataframe(filtered_df)

                # Agregar gráfico de barras para los registros filtrados
                if not filtered_df.empty:
                    st.write("### Gráfico de barras de registros para el año seleccionado")
                    year_count = filtered_df['Fecha'].value_counts().reset_index()
                    year_count.columns = ['Año', 'Cantidad']
                    
                    fig_year = px.bar(year_count, x='Año', y='Cantidad', 
                                      title=f'Número de Registros para el Año {selected_year}',
                                      color='Cantidad', 
                                      labels={'Cantidad': 'Número de Registros'},
                                      color_discrete_sequence=['lightblue'])

                    fig_year.update_traces(marker=dict(line=dict(color='black', width=1.5)))  # Add border color and width

                    st.plotly_chart(fig_year)



# Información adicional
st.sidebar.header("Seminario de Investigación I")
st.sidebar.write("Nombre: Javier Horacio Pérez Ricárdez")
st.sidebar.write("Catedrático: Dr. Jorge Eduardo Brieva Rico")
st.sidebar.write("Fecha: 20 de octubre del 2024")
