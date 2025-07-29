#O objetivvo desse código e fazer a inserção de arquivos JSONs que estão armazenados localmente, no Banco de Dados Postgresql16
#Meu banco contém a tabela bíblia, que possui os campos, id, livro da bíblia, capítulo, versículo e o embedding do texto do versículo


import json
import psycopg2
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "dbname": "biblia", 
    "user": "postgres",           
    "password": "postgres",         
    "host": "localhost",             
    "port": "5432"                  
}

JSON_FILE = r"c:\Users\nicol\OneDrive - Universidade Federal de Uberlândia\UFU\INICIAÇÃO CIENTÍFICA\BANCO DE DADOS VETORIAIS\JSONS BIBLIA\bsb-e5-small-v2-corintias001.json"
TABLE_NAME = "bible_embeddings"

def create_table(conn):
    """Cria a tabela se não existir"""
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            book VARCHAR(50) NOT NULL,
            chapter INT NOT NULL,
            verse INT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            UNIQUE (book, chapter, verse)
        );
        """)
        conn.commit()
        print("✅ Tabela criada com sucesso!")

def validate_data(data):
    """Valida a estrutura do JSON"""
    required_keys = {"book", "chapter", "verse", "embedding"}
    for item in data:
        if not all(key in item for key in required_keys):
            raise ValueError("Estrutura do JSON inválida. Verifique os campos.")
        

def insert_data(conn, data):
    """Insere os dados no PostgreSQL"""
    with conn.cursor() as cur:
        insert_query = f"""
        INSERT INTO {TABLE_NAME} (book, chapter, verse, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (book, chapter, verse) DO NOTHING;
        """
        records = [
            (item["book"], item["chapter"], item["verse"], list(item["embedding"]))
            for item in data
        ]
        execute_batch(cur, insert_query, records, page_size=100)
        conn.commit()
        print(f"✅ {len(records)} embeddings inseridos/atualizados!")

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexão com o PostgreSQL estabelecida!")
        
        with open(JSON_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        validate_data(data)
        create_table(conn)
        insert_data(conn, data)
        
    except psycopg2.Error as e:
        print(f"❌ Erro no PostgreSQL: {e}")
        if 'conn' in locals():
            conn.rollback()
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
