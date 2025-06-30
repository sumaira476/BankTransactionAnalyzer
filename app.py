import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import pdfplumber

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Bank Transaction Analyzer", layout="wide")
st.title("🏦 Bank Transaction Analyzer + Smart Saving Advisor")

# Toggle to show chatbot + Scroll Link
if st.toggle("🤖 Ask BankBuddy (Scroll to Chatbot Below)"):
    st.markdown('<a href="#footer">⬇️ Scroll to Chatbot</a>', unsafe_allow_html=True)

# Sidebar file upload
st.sidebar.header("📁 Upload Your Bank or Wallet Statement")
st.sidebar.markdown("Supported formats: `.csv`, `.xlsx`, `.pdf` (Kotak, SBI, etc.)")
uploaded_file = st.sidebar.file_uploader("⬆️ Upload File Here", type=["csv", "xlsx", "pdf"])

# PDF parsing for Kotak-style statements
def parse_kotak_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        lines = text.split('\n')
        transactions = []

        for line in lines:
            match = re.match(r'(\d{2}-\d{2}-\d{4})\s+(.+?)\s+([\d,]+\.\d{2})\((Cr|Dr)\)', line)
            if match:
                date = pd.to_datetime(match.group(1), dayfirst=True)
                desc = match.group(2).strip()
                amount = float(match.group(3).replace(',', ''))
                if match.group(4) == "Dr":
                    amount = -amount
                transactions.append({"date": date, "description": desc, "amount": amount})
        return pd.DataFrame(transactions)
    except Exception as e:
        st.error(f"❌ PDF parsing failed: {e}")
        return pd.DataFrame()

# Main logic
if uploaded_file:
    with st.spinner("Processing your file..."):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.pdf'):
                df = parse_kotak_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()

            if df.empty or 'date' not in df.columns or 'amount' not in df.columns:
                st.error("❌ Parsed file does not contain usable 'date' and 'amount' columns. Please upload a supported format.")
                st.stop()

            df.columns = [col.lower().strip() for col in df.columns]
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['date', 'amount'])

            if 'category' not in df.columns:
                def auto_category(desc):
                    desc = str(desc).lower()
                    if 'salary' in desc or 'income' in desc:
                        return 'Salary'
                    elif 'rent' in desc:
                        return 'Rent'
                    elif 'food' in desc or 'zomato' in desc or 'swiggy' in desc:
                        return 'Food'
                    elif 'bill' in desc or 'electricity' in desc:
                        return 'Bills'
                    elif 'shopping' in desc or 'amazon' in desc or 'flipkart' in desc:
                        return 'Shopping'
                    elif 'uber' in desc or 'ola' in desc or 'fuel' in desc:
                        return 'Travel'
                    elif 'recharge' in desc:
                        return 'Recharge'
                    elif 'netflix' in desc or 'hotstar' in desc:
                        return 'Entertainment'
                    elif 'pharmacy' in desc or 'hospital' in desc:
                        return 'Medical'
                    elif 'course' in desc or 'education' in desc:
                        return 'Education'
                    else:
                        return 'Other'

                df['description'] = df.get('description', '')
                df['category'] = df['description'].apply(auto_category)

            st.subheader("📊 Expense vs Income Summary")

            income = df[df['amount'] > 0]['amount'].sum()
            expense = -df[df['amount'] < 0]['amount'].sum()
            savings = income - expense

            col1, col2, col3 = st.columns(3)
            col1.metric("Income", f"₹ {income:,.0f}")
            col2.metric("Expenses", f"₹ {expense:,.0f}")
            col3.metric("Savings", f"₹ {savings:,.0f}", delta_color="inverse")

            category_summary = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().reset_index()
            fig = px.pie(category_summary, values='amount', names='category', title='Expenses by Category')
            st.plotly_chart(fig, use_container_width=True)

            df['month'] = df['date'].dt.to_period('M')
            monthly = df.groupby(['month'])['amount'].sum().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            fig2 = px.bar(monthly, x='month', y='amount', title='Net Cash Flow Per Month')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("💡 Smart Saving Recommendations")

            cat_summary = df.groupby('category')['amount'].sum().reset_index()
            cat_summary['percent_of_expense'] = -cat_summary['amount'] / expense * 100
            cat_summary = cat_summary.sort_values(by='percent_of_expense', ascending=False)

            ideal_limits = {
                'Rent': 35, 'Food': 15, 'Shopping': 10, 'Entertainment': 5,
                'Bills': 10, 'Recharge': 2, 'Travel': 8
            }

            recommendations = []
            for _, row in cat_summary.iterrows():
                category = row['category']
                percent = row['percent_of_expense']
                ideal = ideal_limits.get(category, None)
                if ideal and percent > ideal:
                    diff = percent - ideal
                    over_amount = expense * diff / 100
                    recommendations.append(
                        f"⚠️ You spent **{percent:.1f}%** on **{category}** (ideal: {ideal}%). Try reducing by **₹{over_amount:,.0f}**."
                    )

            if recommendations:
                st.markdown("### 🔎 You can save more by:")
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("✅ You're spending within ideal limits. Great job!")

            st.markdown("---")
            st.subheader("📈 Projected Monthly Savings")

            monthly_savings = df.groupby('month')['amount'].sum().reset_index()
            avg_saving = monthly_savings['amount'].mean()
            st.metric("💰 Average Monthly Saving", f"₹ {avg_saving:,.0f}")

            if avg_saving < 5000:
                st.warning("Your savings are low. Aim to save at least ₹5,000/month.")
            else:
                st.success("You're saving a healthy amount. Keep it up!")

            # Footer ID for scroll target
            st.markdown("<div id='footer'></div>", unsafe_allow_html=True)

            # --- Chatbot Section ---
            st.subheader("🤖 BankBuddy AI Assistant")

            user_q = st.text_input("💬 Ask your financial question to BankBuddy:")
            ask_button = st.button("Ask BankBuddy")

            if ask_button and user_q:
                df_sample = df.head(10).to_csv(index=False)
                prompt = f"""
You are BankBuddy, a smart financial assistant.
Here is a sample of user's bank data:
{df_sample}
User asked: {user_q}
Please answer briefly, helpfully, and with tips.
"""
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()
                    st.markdown(f"**BankBuddy says:** {response_text}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("📂 Please upload a bank or wallet statement to get started.")
