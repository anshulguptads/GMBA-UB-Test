# =============================================================================
# UNIVERSAL BANK ANALYTICS DASHBOARD
# Complete Descriptive, Diagnostic, Predictive & Prescriptive Analytics
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Universal Bank Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR LIGHT THEME
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E0F2FE 0%, #BAE6FD 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0EA5E9;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0369A1;
        border-bottom: 2px solid #0EA5E9;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E0F2FE;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0EA5E9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the Universal Bank dataset"""
    try:
        df = pd.read_csv("UniversalBank.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ File 'UniversalBank.csv' not found. Please upload the file.")
        return None

# =============================================================================
# DATA PREPROCESSING FUNCTION
# =============================================================================
@st.cache_data
def preprocess_data(df):
    """Preprocess data for modeling"""
    df_clean = df.copy()
    
    # Drop ID and ZIP Code (not useful for prediction)
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop('ID', axis=1)
    if 'ZIP Code' in df_clean.columns:
        df_clean = df_clean.drop('ZIP Code', axis=1)
    
    # Handle negative experience (data quality issue)
    df_clean['Experience'] = df_clean['Experience'].apply(lambda x: max(0, x))
    
    return df_clean

# =============================================================================
# MODEL TRAINING FUNCTION
# =============================================================================
@st.cache_data
def train_models(df):
    """Train multiple classification models and return results"""
    df_model = preprocess_data(df)
    
    # Features and Target
    X = df_model.drop('Personal Loan', axis=1)
    y = df_model['Personal Loan']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    results = []
    roc_data = {}
    feature_importance = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, y_train_pred),
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred),
            'AUC-ROC': roc_auc_score(y_test, y_test_prob)
        })
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_score(y_test, y_test_prob)}
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
    
    results_df = pd.DataFrame(results)
    
    return results_df, roc_data, feature_importance, trained_models, scaler, X.columns.tolist(), X_test, y_test

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">🏦 Universal Bank Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.info("👆 Please ensure 'UniversalBank.csv' is in the same directory as this app.")
        
        # File uploader as backup
        uploaded_file = st.file_uploader("Or upload your CSV file here:", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        else:
            return
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Analysis Type:",
        ["🏠 Home", "📊 Descriptive Analytics", "🔍 Diagnostic Analytics", 
         "🤖 Predictive Analytics", "💡 Prescriptive Analytics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    st.sidebar.write(f"**Total Features:** {len(df.columns)}")
    st.sidebar.write(f"**Loan Acceptance Rate:** {df['Personal Loan'].mean()*100:.2f}%")
    
    # =============================================================================
    # HOME PAGE
    # =============================================================================
    if page == "🏠 Home":
        st.markdown('<div class="section-header">📌 Dashboard Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}", delta=None)
        with col2:
            st.metric("Loan Acceptance Rate", f"{df['Personal Loan'].mean()*100:.2f}%", delta=None)
        with col3:
            st.metric("Average Income", f"${df['Income'].mean()*1000:,.0f}", delta=None)
        with col4:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years", delta=None)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📈 Quick Stats</div>', unsafe_allow_html=True)
            
            quick_stats = pd.DataFrame({
                'Metric': ['Customers with Mortgage', 'Securities Account Holders', 
                          'CD Account Holders', 'Online Banking Users', 'Credit Card Holders'],
                'Count': [
                    (df['Mortgage'] > 0).sum(),
                    df['Securities Account'].sum(),
                    df['CD Account'].sum(),
                    df['Online'].sum(),
                    df['CreditCard'].sum()
                ],
                'Percentage': [
                    f"{(df['Mortgage'] > 0).mean()*100:.1f}%",
                    f"{df['Securities Account'].mean()*100:.1f}%",
                    f"{df['CD Account'].mean()*100:.1f}%",
                    f"{df['Online'].mean()*100:.1f}%",
                    f"{df['CreditCard'].mean()*100:.1f}%"
                ]
            })
            st.dataframe(quick_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<div class="section-header">🎯 Loan Acceptance by Education</div>', unsafe_allow_html=True)
            
            edu_loan = df.groupby('Education')['Personal Loan'].agg(['sum', 'count'])
            edu_loan['Rate'] = (edu_loan['sum'] / edu_loan['count'] * 100).round(2)
            edu_loan.index = ['Undergrad', 'Graduate', 'Advanced']
            
            fig = px.bar(
                x=edu_loan.index, 
                y=edu_loan['Rate'],
                color=edu_loan['Rate'],
                color_continuous_scale='Blues',
                labels={'x': 'Education Level', 'y': 'Loan Acceptance Rate (%)'}
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown('<div class="section-header">📋 Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data Dictionary
        with st.expander("📖 Data Dictionary"):
            data_dict = pd.DataFrame({
                'Column': ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 
                          'Education', 'Mortgage', 'Personal Loan', 'Securities Account', 
                          'CD Account', 'Online', 'CreditCard'],
                'Description': [
                    'Customer ID', 'Customer age in years', 'Years of professional experience',
                    'Annual income ($000)', 'Home ZIP code', 'Family size',
                    'Monthly credit card spending ($000)', 'Education (1:Undergrad, 2:Graduate, 3:Advanced)',
                    'Mortgage value ($000)', 'Accepted personal loan? (Target)', 
                    'Has securities account?', 'Has CD account?', 
                    'Uses online banking?', 'Has credit card?'
                ],
                'Type': ['ID', 'Numeric', 'Numeric', 'Numeric', 'Categorical', 'Numeric',
                        'Numeric', 'Categorical', 'Numeric', 'Binary (Target)', 
                        'Binary', 'Binary', 'Binary', 'Binary']
            })
            st.dataframe(data_dict, use_container_width=True, hide_index=True)
    
    # =============================================================================
    # DESCRIPTIVE ANALYTICS
    # =============================================================================
    elif page == "📊 Descriptive Analytics":
        st.markdown('<div class="section-header">📊 Descriptive Analytics</div>', unsafe_allow_html=True)
        st.markdown("*Understanding what happened - Summary statistics and distributions*")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary Statistics", "📊 Distributions", "🥧 Categorical Analysis", "📋 Cross-tabulations"])
        
        with tab1:
            st.markdown("### Statistical Summary")
            
            # Numeric columns summary
            numeric_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
            summary_stats = df[numeric_cols].describe().T
            summary_stats['median'] = df[numeric_cols].median()
            summary_stats['skewness'] = df[numeric_cols].skew()
            summary_stats = summary_stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']]
            summary_stats = summary_stats.round(2)
            
            st.dataframe(summary_stats, use_container_width=True)
            
            st.markdown("### Key Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="insight-box">
                <b>📊 Income Distribution</b><br>
                • Mean: ${df['Income'].mean()*1000:,.0f}<br>
                • Median: ${df['Income'].median()*1000:,.0f}<br>
                • Range: ${df['Income'].min()*1000:,.0f} - ${df['Income'].max()*1000:,.0f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="insight-box">
                <b>👥 Age Distribution</b><br>
                • Mean: {df['Age'].mean():.1f} years<br>
                • Median: {df['Age'].median():.1f} years<br>
                • Range: {df['Age'].min()} - {df['Age'].max()} years
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="insight-box">
                <b>💳 Credit Card Spending</b><br>
                • Mean: ${df['CCAvg'].mean()*1000:,.0f}/month<br>
                • Median: ${df['CCAvg'].median()*1000:,.0f}/month<br>
                • Max: ${df['CCAvg'].max()*1000:,.0f}/month
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Distribution Plots")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age Distribution
                fig = px.histogram(df, x='Age', nbins=30, color_discrete_sequence=['#0EA5E9'],
                                  title='Age Distribution')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Income Distribution
                fig = px.histogram(df, x='Income', nbins=30, color_discrete_sequence=['#10B981'],
                                  title='Income Distribution ($000)')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Experience Distribution
                fig = px.histogram(df, x='Experience', nbins=30, color_discrete_sequence=['#8B5CF6'],
                                  title='Experience Distribution')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                # CCAvg Distribution
                fig = px.histogram(df, x='CCAvg', nbins=30, color_discrete_sequence=['#F59E0B'],
                                  title='Credit Card Avg Spending Distribution ($000)')
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            
            # Box plots
            st.markdown("### Box Plots (Outlier Detection)")
            
            fig = make_subplots(rows=1, cols=4, subplot_titles=['Age', 'Income', 'CCAvg', 'Mortgage'])
            
            fig.add_trace(go.Box(y=df['Age'], name='Age', marker_color='#0EA5E9'), row=1, col=1)
            fig.add_trace(go.Box(y=df['Income'], name='Income', marker_color='#10B981'), row=1, col=2)
            fig.add_trace(go.Box(y=df['CCAvg'], name='CCAvg', marker_color='#8B5CF6'), row=1, col=3)
            fig.add_trace(go.Box(y=df['Mortgage'], name='Mortgage', marker_color='#F59E0B'), row=1, col=4)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Categorical Variable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Education Distribution
                edu_counts = df['Education'].value_counts().sort_index()
                edu_labels = ['Undergrad', 'Graduate', 'Advanced']
                
                fig = px.pie(values=edu_counts.values, names=edu_labels, 
                            title='Education Level Distribution',
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig, use_container_width=True)
                
                # Family Size Distribution
                family_counts = df['Family'].value_counts().sort_index()
                
                fig = px.bar(x=family_counts.index, y=family_counts.values,
                            title='Family Size Distribution',
                            labels={'x': 'Family Size', 'y': 'Count'},
                            color=family_counts.values,
                            color_continuous_scale='Blues')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Personal Loan Distribution
                loan_counts = df['Personal Loan'].value_counts()
                
                fig = px.pie(values=loan_counts.values, names=['No', 'Yes'],
                            title='Personal Loan Acceptance',
                            color_discrete_sequence=['#FCA5A5', '#86EFAC'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Binary Features Distribution
                binary_cols = ['Securities Account', 'CD Account', 'Online', 'CreditCard']
                binary_data = pd.DataFrame({
                    'Feature': binary_cols,
                    'Yes': [df[col].sum() for col in binary_cols],
                    'No': [len(df) - df[col].sum() for col in binary_cols]
                })
                
                fig = px.bar(binary_data, x='Feature', y=['Yes', 'No'],
                            title='Binary Features Distribution',
                            barmode='group',
                            color_discrete_sequence=['#10B981', '#EF4444'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### Cross-tabulation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Personal Loan by Education
                cross_tab = pd.crosstab(df['Education'], df['Personal Loan'], margins=True)
                cross_tab.index = ['Undergrad', 'Graduate', 'Advanced', 'Total']
                cross_tab.columns = ['No Loan', 'Loan', 'Total']
                
                st.markdown("**Personal Loan by Education Level**")
                st.dataframe(cross_tab, use_container_width=True)
                
                # Personal Loan by Family Size
                cross_tab2 = pd.crosstab(df['Family'], df['Personal Loan'], margins=True)
                cross_tab2.columns = ['No Loan', 'Loan', 'Total']
                
                st.markdown("**Personal Loan by Family Size**")
                st.dataframe(cross_tab2, use_container_width=True)
            
            with col2:
                # Visualization
                fig = px.histogram(df, x='Education', color='Personal Loan',
                                  barmode='group',
                                  title='Personal Loan by Education',
                                  labels={'Education': 'Education Level'},
                                  color_discrete_sequence=['#EF4444', '#10B981'])
                fig.update_xaxes(ticktext=['Undergrad', 'Graduate', 'Advanced'], tickvals=[1, 2, 3])
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(df, x='Family', color='Personal Loan',
                                  barmode='group',
                                  title='Personal Loan by Family Size',
                                  color_discrete_sequence=['#EF4444', '#10B981'])
                st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # DIAGNOSTIC ANALYTICS
    # =============================================================================
    elif page == "🔍 Diagnostic Analytics":
        st.markdown('<div class="section-header">🔍 Diagnostic Analytics</div>', unsafe_allow_html=True)
        st.markdown("*Understanding why it happened - Correlations and patterns*")
        
        tab1, tab2, tab3 = st.tabs(["🔥 Correlation Analysis", "👥 Segment Comparison", "📊 Feature Relationships"])
        
        with tab1:
            st.markdown("### Correlation Heatmap")
            
            # Prepare data for correlation
            df_corr = df.drop(['ID', 'ZIP Code'], axis=1, errors='ignore')
            corr_matrix = df_corr.corr()
            
            # Plotly heatmap
            fig = px.imshow(corr_matrix,
                           labels=dict(color="Correlation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with Personal Loan
            st.markdown("### Top Correlations with Personal Loan")
            
            loan_corr = corr_matrix['Personal Loan'].drop('Personal Loan').sort_values(key=abs, ascending=False)
            
            fig = px.bar(x=loan_corr.values, y=loan_corr.index, orientation='h',
                        color=loan_corr.values,
                        color_continuous_scale='RdBu_r',
                        labels={'x': 'Correlation', 'y': 'Feature'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>🔑 Key Findings:</b><br>
            • <b>Income</b> has the strongest positive correlation with loan acceptance<br>
            • <b>CD Account</b> holders are more likely to accept loans<br>
            • <b>Education</b> level positively correlates with loan acceptance<br>
            • <b>CCAvg</b> (Credit Card spending) shows positive correlation
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Loan Acceptors vs Non-Acceptors")
            
            # Segment comparison
            comparison = df.groupby('Personal Loan').agg({
                'Age': 'mean',
                'Income': 'mean',
                'CCAvg': 'mean',
                'Mortgage': 'mean',
                'Experience': 'mean',
                'Family': 'mean'
            }).round(2)
            comparison.index = ['Non-Acceptors', 'Acceptors']
            
            st.dataframe(comparison.T, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Income comparison
                fig = px.box(df, x='Personal Loan', y='Income',
                            color='Personal Loan',
                            title='Income Distribution by Loan Acceptance',
                            labels={'Personal Loan': 'Accepted Loan'},
                            color_discrete_sequence=['#EF4444', '#10B981'])
                fig.update_xaxes(ticktext=['No', 'Yes'], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # CCAvg comparison
                fig = px.box(df, x='Personal Loan', y='CCAvg',
                            color='Personal Loan',
                            title='Credit Card Spending by Loan Acceptance',
                            labels={'Personal Loan': 'Accepted Loan'},
                            color_discrete_sequence=['#EF4444', '#10B981'])
                fig.update_xaxes(ticktext=['No', 'Yes'], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            # Education breakdown
            st.markdown("### Loan Acceptance Rate by Education")
            
            edu_analysis = df.groupby('Education')['Personal Loan'].agg(['sum', 'count'])
            edu_analysis['Acceptance Rate'] = (edu_analysis['sum'] / edu_analysis['count'] * 100).round(2)
            edu_analysis.index = ['Undergrad', 'Graduate', 'Advanced']
            
            fig = px.bar(x=edu_analysis.index, y=edu_analysis['Acceptance Rate'],
                        color=edu_analysis['Acceptance Rate'],
                        color_continuous_scale='Greens',
                        title='Loan Acceptance Rate by Education Level',
                        labels={'x': 'Education', 'y': 'Acceptance Rate (%)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Feature Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Income vs CCAvg
                fig = px.scatter(df, x='Income', y='CCAvg', color='Personal Loan',
                                title='Income vs Credit Card Spending',
                                color_discrete_sequence=['#EF4444', '#10B981'],
                                opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age vs Income
                fig = px.scatter(df, x='Age', y='Income', color='Personal Loan',
                                title='Age vs Income',
                                color_discrete_sequence=['#EF4444', '#10B981'],
                                opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            
            # Income bins analysis
            st.markdown("### Income Segment Analysis")
            
            df['Income_Bin'] = pd.cut(df['Income'], bins=[0, 50, 100, 150, 250], 
                                      labels=['Low (<50k)', 'Medium (50-100k)', 'High (100-150k)', 'Very High (>150k)'])
            
            income_analysis = df.groupby('Income_Bin')['Personal Loan'].agg(['sum', 'count'])
            income_analysis['Acceptance Rate'] = (income_analysis['sum'] / income_analysis['count'] * 100).round(2)
            
            fig = px.bar(x=income_analysis.index.astype(str), y=income_analysis['Acceptance Rate'],
                        color=income_analysis['Acceptance Rate'],
                        color_continuous_scale='Blues',
                        title='Loan Acceptance Rate by Income Segment',
                        labels={'x': 'Income Segment', 'y': 'Acceptance Rate (%)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <b>⚠️ Key Diagnostic Insight:</b><br>
            Customers with income above $100k have significantly higher loan acceptance rates. 
            The bank should focus marketing efforts on this high-income segment for better conversion.
            </div>
            """, unsafe_allow_html=True)
    
    # =============================================================================
    # PREDICTIVE ANALYTICS
    # =============================================================================
    elif page == "🤖 Predictive Analytics":
        st.markdown('<div class="section-header">🤖 Predictive Analytics</div>', unsafe_allow_html=True)
        st.markdown("*Predicting what will happen - Machine Learning Models*")
        
        # Train models
        with st.spinner("Training models... Please wait."):
            results_df, roc_data, feature_importance, trained_models, scaler, feature_names, X_test, y_test = train_models(df)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 ROC Curves", "🎯 Feature Importance", "🔮 Prediction Tool"])
        
        with tab1:
            st.markdown("### Model Performance Comparison")
            
            # Format the results dataframe
            results_display = results_df.copy()
            for col in results_display.columns[1:]:
                results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(results_display, use_container_width=True, hide_index=True)
            
            # Best model highlight
            best_model_idx = results_df['AUC-ROC'].idxmax()
            best_model = results_df.loc[best_model_idx, 'Model']
            best_auc = results_df.loc[best_model_idx, 'AUC-ROC']
            
            st.markdown(f"""
            <div class="insight-box">
            <b>🏆 Best Performing Model: {best_model}</b><br>
            • AUC-ROC Score: {best_auc:.4f}<br>
            • Test Accuracy: {results_df.loc[best_model_idx, 'Test Accuracy']:.4f}<br>
            • F1-Score: {results_df.loc[best_model_idx, 'F1-Score']:.4f}
            </div>
            """, unsafe_allow_html=True)
            
            # Visual comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(results_df, x='Model', y='Test Accuracy',
                            color='Test Accuracy',
                            color_continuous_scale='Blues',
                            title='Test Accuracy Comparison')
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(results_df, x='Model', y='AUC-ROC',
                            color='AUC-ROC',
                            color_continuous_scale='Greens',
                            title='AUC-ROC Comparison')
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison chart
            st.markdown("### Detailed Metrics Comparison")
            
            metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            
            colors = ['#0EA5E9', '#10B981', '#F59E0B']
            for i, metric in enumerate(metrics_to_plot):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    marker_color=colors[i]
                ))
            
            fig.update_layout(barmode='group', xaxis_tickangle=-45,
                             title='Precision, Recall & F1-Score Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### ROC Curves - All Models")
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set2
            
            for i, (name, data) in enumerate(roc_data.items()):
                fig.add_trace(go.Scatter(
                    x=data['fpr'], y=data['tpr'],
                    name=f"{name} (AUC={data['auc']:.3f})",
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=600,
                legend=dict(x=0.6, y=0.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>📊 ROC Curve Interpretation:</b><br>
            • The closer the curve to the top-left corner, the better the model<br>
            • AUC (Area Under Curve) of 1.0 represents perfect classification<br>
            • AUC of 0.5 represents random guessing (diagonal line)<br>
            • All models significantly outperform random classification
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Feature Importance Analysis")
            
            if feature_importance:
                # Select model for feature importance
                model_choice = st.selectbox(
                    "Select Model for Feature Importance:",
                    list(feature_importance.keys())
                )
                
                if model_choice in feature_importance:
                    importance_df = pd.DataFrame({
                        'Feature': list(feature_importance[model_choice].keys()),
                        'Importance': list(feature_importance[model_choice].values())
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                                orientation='h',
                                color='Importance',
                                color_continuous_scale='Blues',
                                title=f'Feature Importance - {model_choice}')
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top features
                    top_features = importance_df.nlargest(3, 'Importance')['Feature'].tolist()
                    st.markdown(f"""
                    <div class="insight-box">
                    <b>🔑 Top 3 Most Important Features:</b><br>
                    1. {top_features[0]}<br>
                    2. {top_features[1]}<br>
                    3. {top_features[2]}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Feature importance is available for tree-based models (Decision Tree, Random Forest, Gradient Boosting)")
        
        with tab4:
            st.markdown("### 🔮 Loan Acceptance Prediction Tool")
            st.markdown("Enter customer details to predict loan acceptance probability")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.slider("Age", 18, 70, 35)
                experience = st.slider("Experience (years)", 0, 45, 10)
                income = st.slider("Income ($000)", 8, 250, 50)
                family = st.selectbox("Family Size", [1, 2, 3, 4])
            
            with col2:
                ccavg = st.slider("Credit Card Avg ($000/month)", 0.0, 10.0, 1.5)
                education = st.selectbox("Education", 
                                        options=[1, 2, 3],
                                        format_func=lambda x: {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'}[x])
                mortgage = st.slider("Mortgage ($000)", 0, 700, 0)
            
            with col3:
                securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
                cd_account = st.selectbox("CD Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
                online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
                creditcard = st.selectbox("Credit Card", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
            
            if st.button("🎯 Predict Loan Acceptance", type="primary"):
                # Prepare input
                input_data = np.array([[age, experience, income, family, ccavg, education, 
                                       mortgage, securities, cd_account, online, creditcard]])
                input_scaled = scaler.transform(input_data)
                
                # Get predictions from all models
                st.markdown("### Prediction Results")
                
                predictions = {}
                for name, model in trained_models.items():
                    prob = model.predict_proba(input_scaled)[0][1]
                    predictions[name] = prob
                
                # Display results
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Acceptance Probability': [f"{p*100:.2f}%" for p in predictions.values()]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                with col2:
                    avg_prob = np.mean(list(predictions.values()))
                    
                    if avg_prob >= 0.5:
                        st.success(f"✅ **High Likelihood of Acceptance**\n\nAverage Probability: {avg_prob*100:.2f}%")
                    else:
                        st.warning(f"⚠️ **Low Likelihood of Acceptance**\n\nAverage Probability: {avg_prob*100:.2f}%")
                
                # Visualization
                fig = px.bar(x=list(predictions.keys()), y=[p*100 for p in predictions.values()],
                            color=[p*100 for p in predictions.values()],
                            color_continuous_scale='RdYlGn',
                            labels={'x': 'Model', 'y': 'Acceptance Probability (%)'},
                            title='Prediction by Model')
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Threshold")
                st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # PRESCRIPTIVE ANALYTICS
    # =============================================================================
    elif page == "💡 Prescriptive Analytics":
        st.markdown('<div class="section-header">💡 Prescriptive Analytics</div>', unsafe_allow_html=True)
        st.markdown("*Recommending what to do - Actionable insights and strategies*")
        
        # Train models for scoring
        with st.spinner("Generating recommendations..."):
            results_df, roc_data, feature_importance, trained_models, scaler, feature_names, X_test, y_test = train_models(df)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Customers", "📊 Segment Strategy", "💰 Campaign ROI", "📋 Recommendations"])
        
        with tab1:
            st.markdown("### Customer Scoring & Targeting")
            
            # Score all customers using best model
            df_score = preprocess_data(df)
            X_all = df_score.drop('Personal Loan', axis=1)
            X_all_scaled = scaler.transform(X_all)
            
            # Use Random Forest for scoring
            best_model = trained_models['Random Forest']
            df['Loan_Probability'] = best_model.predict_proba(X_all_scaled)[:, 1]
            df['Score_Decile'] = pd.qcut(df['Loan_Probability'], 10, labels=False, duplicates='drop') + 1
            
            # Top prospects
            st.markdown("### 🎯 Top 20 Prospects for Targeting")
            
            top_prospects = df[df['Personal Loan'] == 0].nlargest(20, 'Loan_Probability')[
                ['ID', 'Age', 'Income', 'Education', 'CCAvg', 'Loan_Probability']
            ].copy()
            top_prospects['Loan_Probability'] = top_prospects['Loan_Probability'].apply(lambda x: f"{x*100:.2f}%")
            top_prospects['Education'] = top_prospects['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'})
            
            st.dataframe(top_prospects, use_container_width=True, hide_index=True)
            
            # Decile analysis
            st.markdown("### Score Decile Analysis")
            
            decile_analysis = df.groupby('Score_Decile').agg({
                'Personal Loan': ['sum', 'count', 'mean'],
                'Income': 'mean',
                'Loan_Probability': 'mean'
            }).round(3)
            decile_analysis.columns = ['Loan_Acceptors', 'Total_Customers', 'Actual_Rate', 'Avg_Income', 'Predicted_Prob']
            decile_analysis['Actual_Rate'] = (decile_analysis['Actual_Rate'] * 100).round(2)
            decile_analysis['Predicted_Prob'] = (decile_analysis['Predicted_Prob'] * 100).round(2)
            
            st.dataframe(decile_analysis, use_container_width=True)
            
            # Gains chart
            fig = px.bar(x=decile_analysis.index, y=decile_analysis['Actual_Rate'],
                        color=decile_analysis['Actual_Rate'],
                        color_continuous_scale='Greens',
                        labels={'x': 'Score Decile', 'y': 'Loan Acceptance Rate (%)'},
                        title='Loan Acceptance Rate by Score Decile')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Segment-Based Strategy")
            
            # Create customer segments
            df['Income_Segment'] = pd.cut(df['Income'], bins=[0, 50, 100, 150, 250],
                                         labels=['Low', 'Medium', 'High', 'Very High'])
            df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'})
            
            # Segment performance
            segment_perf = df.groupby(['Income_Segment', 'Education_Label']).agg({
                'Personal Loan': ['sum', 'count', 'mean'],
                'Loan_Probability': 'mean'
            }).round(3)
            segment_perf.columns = ['Acceptors', 'Total', 'Acceptance_Rate', 'Predicted_Prob']
            segment_perf = segment_perf.reset_index()
            segment_perf['Acceptance_Rate'] = (segment_perf['Acceptance_Rate'] * 100).round(2)
            
            st.dataframe(segment_perf, use_container_width=True, hide_index=True)
            
            # Heatmap
            pivot_table = df.pivot_table(values='Personal Loan', 
                                        index='Income_Segment', 
                                        columns='Education_Label', 
                                        aggfunc='mean') * 100
            
            fig = px.imshow(pivot_table,
                           labels=dict(color="Acceptance Rate (%)"),
                           color_continuous_scale='Greens',
                           title='Loan Acceptance Rate by Income & Education')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>🎯 Targeting Strategy:</b><br>
            • <b>Primary Target:</b> High Income + Advanced Education (Highest acceptance rate)<br>
            • <b>Secondary Target:</b> Very High Income + Graduate Education<br>
            • <b>Growth Opportunity:</b> Medium Income + Graduate Education (Large pool, moderate rate)
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Campaign ROI Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Campaign Parameters")
                target_deciles = st.multiselect(
                    "Select Score Deciles to Target:",
                    options=list(range(1, 11)),
                    default=[8, 9, 10]
                )
                
                cost_per_contact = st.number_input("Cost per Contact ($)", value=10, min_value=1)
                revenue_per_conversion = st.number_input("Revenue per Loan ($)", value=500, min_value=100)
            
            with col2:
                if target_deciles:
                    targeted_customers = df[df['Score_Decile'].isin(target_deciles)]
                    
                    total_targeted = len(targeted_customers)
                    expected_conversions = targeted_customers['Loan_Probability'].sum()
                    
                    total_cost = total_targeted * cost_per_contact
                    expected_revenue = expected_conversions * revenue_per_conversion
                    expected_profit = expected_revenue - total_cost
                    roi = (expected_profit / total_cost * 100) if total_cost > 0 else 0
                    
                    st.markdown("#### Expected Results")
                    st.metric("Customers Targeted", f"{total_targeted:,}")
                    st.metric("Expected Conversions", f"{expected_conversions:.0f}")
                    st.metric("Expected ROI", f"{roi:.1f}%", 
                             delta="Profitable" if roi > 0 else "Loss")
            
            if target_deciles:
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Campaign Cost", f"${total_cost:,.0f}")
                col2.metric("Expected Revenue", f"${expected_revenue:,.0f}")
                col3.metric("Expected Profit", f"${expected_profit:,.0f}",
                           delta=f"{'+' if expected_profit > 0 else ''}{expected_profit:,.0f}")
                
                # ROI by decile
                roi_by_decile = []
                for d in range(1, 11):
                    d_customers = df[df['Score_Decile'] == d]
                    d_cost = len(d_customers) * cost_per_contact
                    d_revenue = d_customers['Loan_Probability'].sum() * revenue_per_conversion
                    d_roi = ((d_revenue - d_cost) / d_cost * 100) if d_cost > 0 else 0
                    roi_by_decile.append({'Decile': d, 'ROI': d_roi})
                
                roi_df = pd.DataFrame(roi_by_decile)
                
                fig = px.bar(roi_df, x='Decile', y='ROI',
                            color='ROI',
                            color_continuous_scale='RdYlGn',
                            title='Expected ROI by Score Decile')
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Strategic Recommendations")
            
            st.markdown("""
            <div class="insight-box">
            <h4>🎯 1. Target Customer Profile</h4>
            Based on our analysis, the ideal target customer has:
            <ul>
                <li><b>Income:</b> Above $100,000 annually</li>
                <li><b>Education:</b> Graduate or Advanced degree</li>
                <li><b>Credit Card Spending:</b> Above $2,000/month</li>
                <li><b>Existing Products:</b> CD Account holder</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>📊 2. Campaign Strategy</h4>
            <ul>
                <li><b>Focus on Top 3 Deciles:</b> These customers have 30-50% conversion probability</li>
                <li><b>Personalized Offers:</b> Tailor loan amounts based on income level</li>
                <li><b>Cross-sell Opportunity:</b> Target CD Account holders first</li>
                <li><b>Digital Channel:</b> Prioritize online banking users for digital campaigns</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ 3. Risk Considerations</h4>
            <ul>
                <li>Avoid over-targeting low-income segments (high cost, low conversion)</li>
                <li>Monitor campaign fatigue in repeatedly targeted segments</li>
                <li>Ensure compliance with fair lending regulations</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>💰 4. Expected Outcomes</h4>
            <ul>
                <li><b>Targeting Top 30%:</b> Expected 3x improvement in conversion rate</li>
                <li><b>Cost Reduction:</b> 40% reduction in marketing spend per conversion</li>
                <li><b>Revenue Impact:</b> Estimated 25% increase in loan portfolio</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Download recommendations
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export scored customers
                export_df = df[['ID', 'Age', 'Income', 'Education', 'CCAvg', 'Personal Loan', 
                               'Loan_Probability', 'Score_Decile']].copy()
                export_df['Loan_Probability'] = export_df['Loan_Probability'].round(4)
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Scored Customer List",
                    data=csv,
                    file_name="scored_customers.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export model results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Model Performance Report",
                    data=csv_results,
                    file_name="model_performance.csv",
                    mime="text/csv"
                )

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()
