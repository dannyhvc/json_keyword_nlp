from sentence_transformers import SentenceTransformer, util


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    k1 = ["stock_price", "open_time"]
    k2 = ["current_price", "market_open"]

    emb1 = model.encode(k1, convert_to_tensor=True)
    emb2 = model.encode(k2, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb1, emb2)
    
    print("Cosine Similarity Matrix:")
    print(cos_sim)


if __name__ == "__main__":
    main()
