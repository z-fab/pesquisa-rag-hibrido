import os
import sqlite3
import sys

import pandas as pd
import yaml

# --- Configurações ---
DATABASE_PATH = "data/dados.db"  # <--- CAMINHO DO SEU SQLITE
INPUT_YAML = "data/docs.yaml"  # O YAML gerado na etapa anterior
OUTPUT_YAML = "data/docs2.yaml"


class NoAliasDumper(yaml.SafeDumper):
    """Dumper para evitar âncoras e aliases no YAML final, mantendo-o limpo."""

    def ignore_aliases(self, data):
        return True


def get_db_connection(db_path):
    """Estabelece conexão com o SQLite."""
    if not os.path.exists(db_path):
        print(f"[ERRO] Banco de dados não encontrado: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def calculate_column_metrics(series):
    """
    Calcula métricas baseadas no tipo de dado inferido pelo Pandas.
    """
    stats = {}
    total_rows = len(series)

    if total_rows == 0:
        return {"status": "empty_table"}

    # Contagem de Nulos
    null_count = int(series.isnull().sum())
    null_pct = round((null_count / total_rows) * 100, 2)

    stats["row_count"] = total_rows
    stats["null_percentage"] = f"{null_pct}%"

    # Remove nulos para cálculos estatísticos
    clean_series = series.dropna()

    # Se a coluna for numérica (Float/Int)
    if pd.api.types.is_numeric_dtype(series):
        if not clean_series.empty:
            stats.update(
                {
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "mean": float(round(clean_series.mean(), 2)),
                    "std_dev": float(round(clean_series.std(), 2)),
                    # Mediana ajuda a identificar outliers que distorcem a média
                    "median": float(round(clean_series.median(), 2)),
                }
            )

    # Se a coluna for Objeto/Texto/Categórica
    else:
        # Força conversão para string para garantir uniformidade
        clean_series = clean_series.astype(str)
        unique_count = clean_series.nunique()
        stats["unique_values_count"] = unique_count

        # Top 10 valores mais frequentes (Essencial para o LLM entender vocabulário)
        if not clean_series.empty:
            top_counts = clean_series.value_counts().head(10)
            stats["top_frequent_values"] = top_counts.index.tolist()

        # Se cardinalidade for baixa (< 20), lista todos para o LLM ter contexto total
        if 0 < unique_count <= 20:
            stats["all_unique_values"] = clean_series.unique().tolist()

    return stats


def main():
    print("--- Iniciando Enriquecimento de Metadados ---")
    print(f"Lendo template: {INPUT_YAML}")

    try:
        with open(INPUT_YAML, "r", encoding="utf-8") as f:
            schema_data = yaml.safe_load(f)
    except FileNotFoundError:
        print("Arquivo YAML template não encontrado.")
        sys.exit(1)

    conn = get_db_connection(DATABASE_PATH)

    # Obtém lista de tabelas reais no banco para validação
    real_tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()
    print(f"Tabelas encontradas no SQLite: {real_tables}")

    for table_def in schema_data.get("tables", []):
        t_name = table_def["table_name"]

        if t_name not in real_tables:
            print(
                f"[AVISO] Tabela '{t_name}' está no YAML mas NÃO no SQLite. Pulando..."
            )
            table_def["metadata_status"] = "table_not_found_in_db"
            continue

        print(f"Processando tabela: {t_name}...")

        try:
            # Carrega dados reais para DataFrame
            # NOTA: Para tabelas gigantes (>1GB), considere usar chunks ou queries SQL diretas (MIN/MAX/AVG)
            df = pd.read_sql(f"SELECT * FROM {t_name}", conn)

            # Itera sobre as colunas definidas no YAML
            for col_def in table_def.get("columns", []):
                c_name = col_def["name"]

                if c_name in df.columns:
                    metrics = calculate_column_metrics(df[c_name])
                    col_def["statistics"] = metrics
                else:
                    print(f"  -> Coluna '{c_name}' não encontrada na tabela real.")
                    col_def["statistics"] = {"error": "column_missing_in_db"}

        except Exception as e:
            print(f"[ERRO] Falha ao processar tabela {t_name}: {e}")

    conn.close()

    # Salva o arquivo final
    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(
            schema_data,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )

    print(f"\nSucesso! YAML enriquecido salvo em: {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
