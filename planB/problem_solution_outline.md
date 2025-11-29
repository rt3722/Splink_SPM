### **Problem Statement (Plain English)**

Talent data lands in our systems with **random synthetic IDs**. The same person can show up twice with different IDs, and we can’t rely on the ID values to tell us whether two records belong to the same human. Traditional database checks (exact name, exact title, etc.) fail because:

- People write their names differently (`James` vs `Jim`), and job titles vary (“Software Engineer” vs “Developer”).
- Some people truly have the same name and job, so even perfect text matches might refer to *different* individuals unless we check harder facts like phone numbers or emails.

This creates two painful errors:

1. **Fragmented identities (missed duplicates):** We create duplicate profiles because fuzzy similarities aren’t recognized.
2. **Doppelgängers (bad merges):** We accidentally merge two different people because their text looks identical.

We need a repeatable, explainable way to say “these records are the same human” using only the real-world information (names, employers, cities, emails, phone numbers) while ignoring the meaningless IDs.

---

### **Solution Outline (In Simple Terms)**

1. **Prepare the data**
   - Clean up names, titles, employers, cities, emails, and phones so they follow consistent formats (e.g., lowercase emails, normalized phone numbers).
   - Combine the cleaned name/title/employer/location into a short description of the person (a “persona paragraph”).

2. **Turn personas into vectors**
   - Feed each persona paragraph into Google’s Gemini embedding model right inside BigQuery.
   - The model converts the text into a numeric vector that captures meaning (e.g., “Software Engineer at Google” stays close to “Developer at Google”).

3. **Store everything in BigQuery**
   - Keep three BigQuery tables: one for normalized personas/anchors, one for embeddings, and one combined view that the app reads.

4. **When a user picks a talent record**
   - Treat it as a “new arrival” and search BigQuery for potential matches using:
     - **Hard anchors:** exact email/phone matches.
     - **Soft similarity:** closest vectors (e.g., top 50 by cosine distance).
   - Limit results to ~200 candidates so the app stays fast.

5. **Build a small graph in memory**
   - Create a node for the incoming record and nodes for each candidate.
   - Connect nodes when they share an email, share a phone, or have very similar vectors (with different strengths).
   - Because we link candidates to each other as well, we can detect chains like “A matches B, B matches C, so A and C likely relate.”

6. **Score the graph and decide**
   - If the incoming record connects strongly (shared email/phone) to any candidate, flag as an **automatic merge**.
   - If connections are weaker (similar text but no shared anchors), flag for **human review**.
   - If there are no meaningful connections, **insert as a new person**.

7. **Visualize for analysts**
   - Cloud Run app shows the candidate list on the left, the full graph on the right, and the merge/review decision at the bottom so analysts can understand *why* the system decided the way it did.

In short: **clean the data, convert descriptions into machine-friendly vectors, store everything in BigQuery, retrieve likely matches, build a graph to see how the records relate, and use simple rules to decide merge/review/new—then show it clearly in the UI.**

