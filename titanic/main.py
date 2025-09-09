import pandas as pd
import re

# 1. Cargar CSV local
df = pd.read_csv("test.csv")

# 2. Limpiar espacios en columnas de texto
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 3. Función para limpiar y separar nombres en FirstName y LastName
def split_name(name):
    try:
        # Quitar comillas y caracteres extra
        name = re.sub(r'["\'`]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()

        # Separar apellido y resto
        last, rest = name.split(",", 1)
        last = last.strip()
        rest = rest.strip()

        # Si hay paréntesis, usar el nombre real de mujer casada
        if "(" in rest:
            inside = re.findall(r"\((.*?)\)", rest)
            if inside:
                first = inside[0].strip()
                first = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', first)
                return first, last

        # Caso normal: quitar título (Mr., Mrs., Miss, etc.)
        if "." in rest:
            first = rest.split(".", 1)[1].strip()
        else:
            first = rest

        # Limpiar caracteres no deseados de first y last
        first = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', first)
        last = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', last)

        return first, last
    except:
        return name, ""

# 4. Aplicar función al dataset
df[['FirstName', 'LastName']] = df['Name'].apply(lambda x: pd.Series(split_name(x)))

# 5. Imputar valores faltantes en Age con la mediana y redondear
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Age'] = df['Age'].round().astype(int)

# 6. Eliminar columnas irrelevantes
df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Name'], errors='ignore', inplace=True)

# 7. Mostrar primeras filas
print(df.head(20))

# 8. Guardar CSV limpio
df.to_csv("archivo_limpio_final.csv", index=False)
print("CSV limpio con FirstName y LastName listo y guardado como 'archivo_limpio_final.csv'")
