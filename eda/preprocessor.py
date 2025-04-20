from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# =========================
# PREPROCESADOR
# =========================
def build_preprocessor(df, target='Value'):
    print("⏳ Preparando preprocesador...")
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference([target]).tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ])
    return preprocessor, categorical_cols, numerical_cols
