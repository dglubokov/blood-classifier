import os
import sqlite3


CELL_TYPES = {
    'erythropoiesis': {
        'Proerythroblast': 'PEB',
        'Erythroblast': 'EBO',
    },
    'lymphoid': {
        'Immature Lymphocyte': 'LYI',
        'Lymphocyte': 'LYT',
        'Plasma Cell': 'PLM',
    },
    'myeloid': {
        'myeloid_immature': {
            'Myeloblast': 'BLA',
            'Metamyelocyte': 'MMZ',
            'Myelocyte': 'MYB',
            'Promyelocyte': 'PMO',
        },
        'myeloid_mature': {
            'Neutrophil': {
                'Band Neutrophil': 'NGB',
                'Segmented Neutrophil': 'NGS',
            },
            'Basophil': 'BAS',
            'Eosinophil': 'EOS',
            'Monocyte': 'MON',
        },
    },
    'abnormal': {
        'Not Identifiable': 'NIF',
        'Other Cell': 'OTH',
        'Abnormal Eosinophil': 'ABE',
        'Artefact': 'ART',
        'Smudge Cell': 'KSC',
        'Faggott Cell': 'FGC'
    }
}


def create_tables(cur):
    query = """
        CREATE TABLE CellTypes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            code VARCHAR(10)
        )
    """.strip()
    cur.execute(query)

    query = """
        CREATE TABLE Cells (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            image BLOB NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY(type) REFERENCES CellTypes(name)
        )
    """.strip()
    cur.execute(query)

    query = """
        CREATE TABLE ModelsTypes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """.strip()
    cur.execute(query)

    query = """
        CREATE TABLE Models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dump BLOB NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY(type) REFERENCES ModelsTypes(name)
        )
    """.strip()
    cur.execute(query)

    query = """
        CREATE TABLE ModelsToCellTypes (
            id INTEGER PRIMARY KEY,
            id_model INTEGER,
            id_cell_type INTEGER,
            FOREIGN KEY(id_model) REFERENCES Models(id)
            FOREIGN KEY(id_cell_type) REFERENCES CellTypes(id)
        )
    """.strip()
    cur.execute(query)


def main():
    if os.path.isfile('cells.db'):
        os.remove('cells.db')
    con = sqlite3.connect('cells.db')
    cur = con.cursor()
    create_tables(cur)
    con.commit()
    con.close()


main()
