### **Architecture Decision Record: Content-Based Hybrid Entity Resolution**

#### **1. Context**

We ingest high-velocity talent data where incoming profiles often duplicate existing records.

  * **The Problem:** The system uses `ID` and `ACCOUNTID` as **randomly generated synthetic keys**. Consequently, a duplicate record for an existing person will arrive with a **new, unique ID** that has no mathematical relationship to the existing record's ID.
  * **The Goal:** We must identify if the *human entity* represented by `Random_ID_A` is the same as the *human entity* represented by `Random_ID_B`. This requires ignoring the IDs entirely during analysis and relying solely on natural identity data.

-----

### Detailed Problem Description

**The "Identity Gap" in Talent Data**
We are trying to solve two specific failure modes in record linkage that standard database tools cannot handle simultaneously:

**1. The "Fragmented Identity" (Low Recall)**

  * **Scenario:** We have an existing record for `James Smith, Software Eng, Google`.
  * **New Input:** `Jim Smith, Dev, Google Inc`.
  * **Failure:** A standard SQL query `WHERE name = 'Jim Smith'` returns nothing. A strictly rules-based system creates a duplicate record.
  * **Goal:** The system must recognize that "Dev" $\approx$ "Software Eng" and "Jim" $\approx$ "James" via **Vector Similarity**.

**2. The "Doppelgänger" (Low Precision)**

  * **Scenario:** We have `John Smith, Accountant, NYC` (Phone: 555-0100).
  * **New Input:** `John Smith, Accountant, NYC` (Phone: 555-9999).
  * **Failure:** A Vector Search sees 100% text similarity and merges them.
  * **Goal:** The system must recognize that despite high semantic score, the **Distinct Phone Numbers** (Hard Logic) imply these are likely different people (or at least require manual review).

**The Architectural Solution**
We solve this by treating **Similarity** and **Identity** as separate layers.

  * **Similarity (Vector):** "These records *look* alike." (Soft Signal)
  * **Identity (Graph):** "These records *are* connected." (Hard Signal)

By building a graph where nodes are records and edges are weighted signals, we can mathematically determine if the "Soft" and "Hard" signals agree or contradict.

-----

#### **2. Decision**

We will implement a **Content-Only Resolution Pipeline**. The system remains "blind" to IDs during the matching phase. It identifies duplicates exclusively by analyzing observable data: **Semantic Persona** (Vector) and **Contact Evidence** (Graph).

**Technical Stack:**

  * **Storage & Retrieval:** PostgreSQL with `pgvector` (Hybrid Search).
  * **Resolution Engine:** Python with `NetworkX` (In-Memory Graph).

#### **3. Data Strategy (Column Mapping)**

We categorize the input columns into three strict functional roles.

| Role | Purpose | Columns Used |
| :--- | :--- | :--- |
| **A. The Persona (Vector)** | Used to generate the **Semantic Embedding**. These fields describe *who* the person is. Variations here (typos, abbreviations) are handled by the AI model. | `FIRSTNAME`, `LASTNAME`, `MIDDLENAME`, `TITLE`, `CURRENT_EMPLOYER__C`, `MAILINGCITY`, `MAILINGSTATE`, `MAILINGPOSTALCODE` |
| **B. The Anchors (Graph)** | Used as **Hard Linking** evidence. If two records share these exact values, they are strongly connected regardless of their random IDs. | `EMAIL`, `OTHER_EMAIL__C`, `PHONE`, `MOBILEPHONE`, `HOMEPHONE`, `OTHERPHONE` |
| **C. The Payload (Output)** | **Ignored during matching.** We carry these values blindly to the end solely to report *which* random IDs should be merged. | `ID`, `ACCOUNTID` |

#### **4. Blueprint Pseudocode**

**Phase 1: Ingestion (Payload Separation)**
*Goal: Convert the "Persona" columns into numbers and extract the "Anchors".*

```text
FUNCTION ProcessIncomingRecord(input_json):

    // 1. Construct Narrative Blob (Strictly Persona Data)
    blob_string = f"""
        {input_json.FIRSTNAME} {input_json.LASTNAME} 
        is a {input_json.TITLE} 
        working at {input_json.CURRENT_EMPLOYER__C}. 
        Located in {input_json.MAILINGCITY}, {input_json.MAILINGSTATE}.
    """
    
    // 2. Generate Vector
    vector_embedding = GenerateEmbedding(blob_string)
    
    // 3. Extract Anchors (Remove Nulls)
    email_list = [input_json.EMAIL, input_json.OTHER_EMAIL__C]
    phone_list = [input_json.PHONE, input_json.MOBILEPHONE, input_json.HOMEPHONE, input_json.OTHERPHONE]
    
    RETURN vector_embedding, email_list, phone_list
```

**Phase 2: Content-Based Search**
*Goal: Find top 50 matches using ONLY the observable content.*

```text
FUNCTION FindPotentialMatches(new_vector, new_emails, new_phones):

    // Search Strategy:
    // 1. Does the vector look similar? (Fuzzy Match)
    // 2. Do they share an Email? (Exact Match)
    // 3. Do they share a Phone? (Exact Match)
    
    QUERY = """
        SELECT * FROM talent_table
        WHERE 
           (embedding <=> $new_vector) < 0.20
        OR 
           (email IN $new_emails OR other_email IN $new_emails)
        OR 
           (phone IN $new_phones OR mobilephone IN $new_phones ...)
        LIMIT 50
    """
    
    RETURN candidates
```

**Phase 3: Graph Resolution**
*Goal: Connect the dots. If the graph connects them, they are duplicates.*

```text
FUNCTION CheckForDuplicates(new_record, candidates):

    Graph G = new Graph()
    G.add_node("INCOMING")

    FOR candidate IN candidates:
        G.add_node(candidate)

        // Rule A: High Semantic Similarity (Persona Check)
        IF CosineDistance(new_record.vector, candidate.vector) < 0.10:
            G.add_edge("INCOMING", candidate, type="SIMILAR_PROFILE")

        // Rule B: Shared Email (Anchor Check)
        IF Intersect(new_record.emails, candidate.emails):
            G.add_edge("INCOMING", candidate, type="SHARED_EMAIL")

        // Rule C: Shared Phone (Anchor Check)
        IF Intersect(new_record.phones, candidate.phones):
            G.add_edge("INCOMING", candidate, type="SHARED_PHONE")

    // Final Step: See what is connected to "INCOMING"
    connected_cluster = G.get_connected_component("INCOMING")

    IF size(connected_cluster) > 1:
        // RESULT: We found real duplicates based on content.
        // NOW we finally read the Payload IDs to report the merge.
        matched_ids = [c.ID for c in connected_cluster if c != "INCOMING"]
        matched_accounts = [c.ACCOUNTID for c in connected_cluster if c != "INCOMING"]
        
        RETURN "MERGE", matched_ids, matched_accounts
    ELSE:
        RETURN "INSERT NEW"
```




