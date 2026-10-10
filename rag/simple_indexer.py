from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

docs = """
# Company Return & Refund Policy Overview

## 1. Return Window and Eligibility
We offer a 30-day return window. You have 30 calendar days from the exact date of delivery to request a return or exchange. To be eligible for a return, your item must be unused, in the same condition that you received it, and in its original packaging with all original tags attached.

## 2. Non-Returnable and Final Sale Items
Certain types of goods cannot be returned under any circumstances. These include:
- Gift cards and digital download products.
- Personalized or custom-made items.
- Perishable goods such as food or flowers.
- Intimate apparel, swimwear, and unsealed hygiene products.
- Items explicitly marked as "Final Sale" or clearance at the time of purchase.

## 3. Step-by-Step Return Process
To initiate a return, please follow these steps:
1. Log into your account and visit our online returns portal at `store.com/returns`.
2. Enter your Order ID and the email address used during checkout.
3. Select the item(s) you wish to return and choose a reason code.
4. Print out the prepaid return shipping label provided by the system.
5. Pack your items securely in the original packaging, attach the label to the outside, and drop it off at any authorized courier location.

## 4. Refund Methods and Processing Times
Once your returned package is received and inspected at our warehouse:
- **Approval Notification:** You will receive an email confirmation stating whether your return was approved or rejected.
- **Refund Timeline:** Approved refunds are processed within 5 to 10 business days. The funds will automatically be credited back to your original method of payment. 
- **Banking Delays:** Depending on your financial institution, please allow an additional 1 to 2 billing cycles for the credit to officially reflect on your bank or credit card statement.

## 5. Return Shipping Fees and Restocking Charges
- **Customer Remorse:** For standard returns due to change of mind, a flat $5.99 return shipping fee will be deducted from your final refund amount.
- **Defective or Wrong Items:** If you received a damaged, defective, or incorrect item, we will cover 100% of the return shipping costs and issue a full refund without any deductions.
- **Restocking Fee:** Returned items that are missing original tags or show minor signs of handling may be subject to a 15% restocking fee."""

# initialize splitter and create chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([docs])

# initialize a hugging face open source embedding model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# BUILD Vector databases
db = Chroma.from_documents(chunks, embedder, persist_directory="./chroma_db")

print(f"Indexed {len(chunks)} chuks")