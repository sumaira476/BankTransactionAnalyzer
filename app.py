import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
import os
import base64
import re
from io import StringIO
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit page settings
st.set_page_config(page_title="Bank Transaction Analyzer", layout="wide")
st.title("🏦 Bank Transaction Analyzer + Smart Saving Advisor")

# Toggle for chatbot
show_bot = st.toggle("🤖 Ask BankBuddy")
user_q = ""
response_text = ""

st.sidebar.header("📁 Upload Your Bank or Wallet Statement")
st.sidebar.markdown("Supported formats: `.csv`, `.xlsx`, `.pdf`, PhonePe, GPay, Paytm, ICICI, HDFC, SBI, Axis, Kotak and more.")

uploaded_file = st.sidebar.file_uploader("⬆️ Upload File Here", type=["csv", "xlsx", "pdf"])

def parse_kotak_pdf(pdf_file):
    text = extract_text(pdf_file)
    lines = text.splitlines()
    data = []
    pattern = r"(\d{2}-[A-Za-z]{3}-\d{4})\s+(.+?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            date, desc, debit, credit, balance = match.groups()
            amt = 0
            if credit:
                amt = float(credit.replace(",", ""))
            elif debit:
                amt = -float(debit.replace(",", ""))
            data.append({
                "date": pd.to_datetime(date, errors='coerce'),
                "description": desc.strip(),
                "amount": amt
            })

    return pd.DataFrame(data)

def generate_pdf(income, expense, savings, recommendations, investment_ideas):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Bank Transaction Summary Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Income: Rs. {income:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Expense: Rs. {expense:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Savings: Rs. {savings:,.0f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    for rec in recommendations:
        pdf.multi_cell(0, 10, txt=rec.encode('latin-1', errors='ignore').decode('latin-1'))
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Investment Suggestions:", ln=True)
    pdf.set_font("Arial", size=12)
    for idea in investment_ideas:
        pdf.multi_cell(0, 10, txt=idea.encode('latin-1', errors='ignore').decode('latin-1'))
    pdf_output = pdf.output(dest='S').encode('latin-1', errors='ignore')
    return pdf_output

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

            st.success("✅ File uploaded and read successfully.")

            df.columns = [col.lower().strip() for col in df.columns]
            if 'date' not in df.columns or 'amount' not in df.columns:
                st.error("❌ File must contain at least 'date' and 'amount' columns.")
                st.stop()

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
                        f"⚠️ You spent {percent:.1f}% on {category} (ideal: {ideal}%). Try reducing by ₹{over_amount:,.0f}.")

            if recommendations:
                st.markdown("### 🔎 You can save more by:")
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("✅ You're spending within ideal limits. Great job!")

            investment_ideas = [
                "Start a SIP in mutual funds with at least ₹1000/month.",
                "Use a recurring deposit to lock monthly savings.",
                "Avoid unnecessary subscriptions and impulse buys.",
                "Track credit card spending limits and set alerts."
            ]

            st.markdown("---")
            st.subheader("📈 Projected Monthly Savings")

            monthly_savings = df.groupby('month')['amount'].sum().reset_index()
            avg_saving = monthly_savings['amount'].mean()
            st.metric("💰 Average Monthly Saving", f"₹ {avg_saving:,.0f}")

            if avg_saving < 5000:
                st.warning("Your savings are low. Aim to save at least ₹5,000/month.")
            else:
                st.success("You're saving a healthy amount. Keep it up!")

            st.subheader("📤 Export Report")
            if st.button("📄 Generate PDF Report"):
                try:
                    pdf_bytes = generate_pdf(income, expense, savings, recommendations, investment_ideas)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="smart_saving_report.pdf">Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Something went wrong while generating PDF: {e}")

            if show_bot:
                st.subheader("💬 Ask BankBuddy")
                user_q = st.text_input("Type your question to BankBuddy")
                ask_button = st.button("Ask")

                if ask_button and user_q:
                    df_sample = df[['category', 'amount']].head(5).to_csv(index=False)
                    prompt = f"""
You are BankBuddy, a smart financial assistant.
Below is a small sample of the user's recent transactions:
{df_sample}
User asked: "{user_q}"
Provide a helpful, friendly, and concise answer.
"""
                    try:
                        with st.spinner("💭 BankBuddy is thinking..."):
                            response = model.generate_content(prompt)
                            response_text = response.text.strip()
                            st.markdown(f"**BankBuddy says:** {response_text}")
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("📂 Please upload a bank or wallet statement from the sidebar to get started.")
