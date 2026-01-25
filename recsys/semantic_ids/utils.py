def prepare_data(df):
    # 1. Prepare Data
    print("Encoding Text...")
    texts = []
    for _, row in df.iterrows():
        t = str(row['title'])
        genres = str(row['genres'])
        # genres = ', '.join(map(str, row['genres']))  # Convert list to string
        blob = f"{t}. {genres}. {row.get('description', '')}"
        texts.append(blob)
    return texts