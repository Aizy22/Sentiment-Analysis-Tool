import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import zipfile

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
    
    .neutral-sentiment {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: #333;
        margin: 0.2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.headers = {"Authorization": f"Bearer {st.secrets.get('HUGGINGFACE_API_KEY', 'YOUR_API_KEY_HERE')}"}
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text"""
        try:
            payload = {"inputs": text}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    scores = result[0]
                    
                    # Convert to readable format
                    sentiment_mapping = {
                        'LABEL_0': 'Negative',
                        'LABEL_1': 'Neutral', 
                        'LABEL_2': 'Positive'
                    }
                    
                    processed_scores = []
                    for item in scores:
                        processed_scores.append({
                            'label': sentiment_mapping.get(item['label'], item['label']),
                            'score': item['score']
                        })
                    
                    # Get primary sentiment
                    primary_sentiment = max(processed_scores, key=lambda x: x['score'])
                    
                    return {
                        'primary_sentiment': primary_sentiment['label'],
                        'confidence': primary_sentiment['score'],
                        'all_scores': processed_scores,
                        'keywords': self.extract_keywords(text)
                    }
            
            return self.fallback_analysis(text)
            
        except Exception as e:
            st.warning(f"API Error: {str(e)}. Using fallback analysis.")
            return self.fallback_analysis(text)
    
    def fallback_analysis(self, text):
        """Simple rule-based fallback analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'awesome', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing', 'poor', 'useless']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'Positive'
            confidence = min(0.6 + (pos_count * 0.1), 0.95)
        elif neg_count > pos_count:
            sentiment = 'Negative'
            confidence = min(0.6 + (neg_count * 0.1), 0.95)
        else:
            sentiment = 'Neutral'
            confidence = 0.6
        
        return {
            'primary_sentiment': sentiment,
            'confidence': confidence,
            'all_scores': [
                {'label': sentiment, 'score': confidence},
                {'label': 'Neutral' if sentiment != 'Neutral' else 'Positive', 'score': (1-confidence)/2},
                {'label': 'Negative' if sentiment == 'Positive' else 'Positive', 'score': (1-confidence)/2}
            ],
            'keywords': self.extract_keywords(text)
        }
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get most common words
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(5)]

    def batch_analyze(self, texts):
        """Analyze multiple texts"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, text in enumerate(texts):
            status_text.text(f'Analyzing text {i+1} of {len(texts)}...')
            result = self.analyze_sentiment(text)
            result['text'] = text[:100] + '...' if len(text) > 100 else text
            result['timestamp'] = datetime.now()
            results.append(result)
            
            progress_bar.progress((i + 1) / len(texts))
            time.sleep(0.1)  # Small delay to avoid overwhelming the API
        
        status_text.text('Analysis complete!')
        return results

def create_sentiment_distribution_chart(results):
    """Create a pie chart showing sentiment distribution"""
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    for result in results:
        sentiment_counts[result['primary_sentiment']] += 1
    
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#00f2fe',
            'Negative': '#fa709a', 
            'Neutral': '#a8edea'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_confidence_chart(results):
    """Create a bar chart showing confidence levels"""
    sentiments = [result['primary_sentiment'] for result in results]
    confidences = [result['confidence'] for result in results]
    texts = [result['text'] for result in results]
    
    colors = ['#00f2fe' if s == 'Positive' else '#fa709a' if s == 'Negative' else '#a8edea' for s in sentiments]
    
    # Convert range to list for Plotly compatibility
    x_values = list(range(len(confidences)))
    
    fig = go.Figure(data=[
        go.Bar(x=x_values, y=confidences, 
               marker_color=colors, text=sentiments, textposition='auto',
               hovertemplate='<b>Text:</b> %{customdata}<br><b>Sentiment:</b> %{text}<br><b>Confidence:</b> %{y:.2%}<extra></extra>',
               customdata=texts)
    ])
    
    fig.update_layout(
        title="Confidence Scores by Text",
        xaxis_title="Text Index",
        yaxis_title="Confidence Score",
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def create_time_series_chart(results):
    """Create a time series chart for sentiment trends"""
    if len(results) < 2:
        return None
    
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create rolling sentiment score (positive=1, neutral=0, negative=-1)
    sentiment_scores = []
    for _, row in df.iterrows():
        if row['primary_sentiment'] == 'Positive':
            sentiment_scores.append(row['confidence'])
        elif row['primary_sentiment'] == 'Negative':
            sentiment_scores.append(-row['confidence'])
        else:
            sentiment_scores.append(0)
    
    df['sentiment_score'] = sentiment_scores
    
    fig = px.line(df, x='timestamp', y='sentiment_score', 
                  title='Sentiment Trend Over Time',
                  labels={'sentiment_score': 'Sentiment Score', 'timestamp': 'Time'})
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    
    return fig

def create_word_cloud(results):
    """Create a word cloud from all keywords"""
    all_keywords = []
    for result in results:
        all_keywords.extend(result['keywords'])
    
    if not all_keywords:
        return None
    
    keyword_text = ' '.join(all_keywords)
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(keyword_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title('Keywords Word Cloud', fontsize=16, pad=20)
    
    return fig

def export_to_csv(results):
    """Export results to CSV"""
    data = []
    for result in results:
        data.append({
            'Text': result['text'],
            'Primary_Sentiment': result['primary_sentiment'],
            'Confidence': result['confidence'],
            'Keywords': ', '.join(result['keywords']),
            'Timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def export_to_json(results):
    """Export results to JSON"""
    export_data = []
    for result in results:
        export_data.append({
            'text': result['text'],
            'primary_sentiment': result['primary_sentiment'],
            'confidence': result['confidence'],
            'all_scores': result['all_scores'],
            'keywords': result['keywords'],
            'timestamp': result['timestamp'].isoformat()
        })
    
    return json.dumps(export_data, indent=2)

def create_pdf_report(results):
    """Create a PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Sentiment Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Summary
    total_texts = len(results)
    positive_count = sum(1 for r in results if r['primary_sentiment'] == 'Positive')
    negative_count = sum(1 for r in results if r['primary_sentiment'] == 'Negative')
    neutral_count = sum(1 for r in results if r['primary_sentiment'] == 'Neutral')
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    summary_text = f"""
    <b>Analysis Summary</b><br/>
    Total Texts Analyzed: {total_texts}<br/>
    Positive Sentiments: {positive_count} ({positive_count/total_texts*100:.1f}%)<br/>
    Negative Sentiments: {negative_count} ({negative_count/total_texts*100:.1f}%)<br/>
    Neutral Sentiments: {neutral_count} ({neutral_count/total_texts*100:.1f}%)<br/>
    Average Confidence: {avg_confidence:.2%}<br/><br/>
    """
    
    summary = Paragraph(summary_text, styles['Normal'])
    story.append(summary)
    story.append(Spacer(1, 20))
    
    # Detailed Results Table
    table_data = [['Text', 'Sentiment', 'Confidence', 'Keywords']]
    for result in results[:10]:  # Limit to first 10 results
        table_data.append([
            result['text'][:50] + '...' if len(result['text']) > 50 else result['text'],
            result['primary_sentiment'],
            f"{result['confidence']:.2%}",
            ', '.join(result['keywords'][:3])
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Interactive Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Analyze emotional tone in text data with advanced NLP and beautiful visualizations")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("Hugging Face API Key", type="password", 
                               help="Enter your Hugging Face API key for enhanced analysis")
        if api_key:
            st.session_state.analyzer.headers["Authorization"] = f"Bearer {api_key}"
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("Analysis Options")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_keywords = st.checkbox("Show extracted keywords", value=True)
        batch_size = st.slider("Batch processing size", 1, 10, 5)
        
        st.markdown("---")
        
        # Export options
        st.subheader("📥 Export Results")
        if st.session_state.results:
            st.write("**Download your analysis results:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = export_to_csv(st.session_state.results)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download results as CSV file"
                )
            
            with col2:
                json_data = export_to_json(st.session_state.results)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download results as JSON file"
                )
            
            with col3:
                try:
                    pdf_data = create_pdf_report(st.session_state.results)
                    st.download_button(
                        label="📑 Download PDF",
                        data=pdf_data,
                        file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download professional PDF report"
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {str(e)}")
                    st.info("CSV and JSON exports are still available above")
        else:
            st.info("📊 Analyze some text first to enable exports")
            st.write("Once you analyze text, you'll be able to download:")
            st.write("• **CSV** - Spreadsheet format for data analysis")
            st.write("• **JSON** - Structured data for developers") 
            st.write("• **PDF** - Professional report for presentations")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Text Analysis", "📊 Batch Analysis", "📈 Analytics Dashboard", "📥 Export Results", "📋 Results History"])
    
    with tab1:
        st.subheader("Single Text Analysis")
        
        # Initialize session state for clearing
        if 'clear_single_text' not in st.session_state:
            st.session_state.clear_single_text = False
        
        # Text input with clear functionality
        if st.session_state.clear_single_text:
            text_input = ""
            st.session_state.clear_single_text = False
        else:
            text_input = st.text_area("Enter text to analyze:", 
                                     placeholder="Type or paste your text here...",
                                     height=150,
                                     key="single_text_input")
        
        # Force clear the text area if needed
        if st.session_state.clear_single_text:
            st.text_area("Enter text to analyze:", 
                        value="",
                        placeholder="Type or paste your text here...",
                        height=150,
                        key="single_text_input_cleared")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Analyze Text", type="primary"):
                if text_input and text_input.strip():
                    with st.spinner("Analyzing sentiment..."):
                        result = st.session_state.analyzer.analyze_sentiment(text_input)
                        result['text'] = text_input
                        result['timestamp'] = datetime.now()
                        st.session_state.results.append(result)
                        
                        # Display result
                        sentiment = result['primary_sentiment']
                        confidence = result['confidence']
                        
                        if sentiment == 'Positive':
                            st.markdown(f'<div class="positive-sentiment"><h3>😊 Positive Sentiment</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                        elif sentiment == 'Negative':
                            st.markdown(f'<div class="negative-sentiment"><h3>😔 Negative Sentiment</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="neutral-sentiment"><h3>😐 Neutral Sentiment</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                        
                        if show_confidence:
                            st.write("**Detailed Scores:**")
                            for score in result['all_scores']:
                                st.write(f"- {score['label']}: {score['score']:.2%}")
                        
                        if show_keywords and result['keywords']:
                            st.write("**Key Words:**", ', '.join(result['keywords']))
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            if st.button("🧹 Clear Text"):
                st.session_state.clear_single_text = True
                # Clear the session state key for the text area
                if "single_text_input" in st.session_state:
                    del st.session_state["single_text_input"]
                st.rerun()
    
    with tab2:
        st.subheader("Batch Text Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'csv'])
        
        # Multiple text inputs
        st.write("**Or enter multiple texts manually:**")
        num_texts = st.number_input("Number of texts", min_value=1, max_value=20, value=3)
        
        # Initialize session state for batch clearing
        if 'clear_batch_texts' not in st.session_state:
            st.session_state.clear_batch_texts = False
        
        texts = []
        for i in range(num_texts):
            # Create unique key for each text input
            input_key = f"batch_text_{i}_{st.session_state.get('batch_clear_counter', 0)}"
            text = st.text_input(f"Text {i+1}:", key=input_key)
            if text.strip():
                texts.append(text)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Analyze All", type="primary"):
                all_texts = texts.copy()
                
                # Process uploaded file
                if uploaded_file:
                    file_content = str(uploaded_file.read(), "utf-8")
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(io.StringIO(file_content))
                        # Assume first column contains text
                        all_texts.extend(df.iloc[:, 0].astype(str).tolist())
                    else:
                        # Split by lines for txt files
                        file_texts = [line.strip() for line in file_content.split('\n') if line.strip()]
                        all_texts.extend(file_texts)
                
                if all_texts:
                    batch_results = st.session_state.analyzer.batch_analyze(all_texts)
                    st.session_state.results.extend(batch_results)
                    
                    st.success(f"✅ Analyzed {len(batch_results)} texts successfully!")
                    
                    # Quick summary
                    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                    for result in batch_results:
                        sentiment_counts[result['primary_sentiment']] += 1
                    
                    col_pos, col_neg, col_neu = st.columns(3)
                    with col_pos:
                        st.metric("😊 Positive", sentiment_counts['Positive'])
                    with col_neg:
                        st.metric("😔 Negative", sentiment_counts['Negative'])
                    with col_neu:
                        st.metric("😐 Neutral", sentiment_counts['Neutral'])
                else:
                    st.warning("Please enter texts or upload a file.")
        
        with col2:
            if st.button("🧹 Clear All"):
                # Increment counter to force new keys for all inputs
                st.session_state.batch_clear_counter = st.session_state.get('batch_clear_counter', 0) + 1
                st.rerun()
    
    with tab3:
        st.subheader("Analytics Dashboard")
        
        if not st.session_state.results:
            st.info("👆 Analyze some text first to see analytics!")
        else:
            # Summary metrics
            total_results = len(st.session_state.results)
            positive_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Positive')
            negative_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Negative')
            neutral_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Neutral')
            avg_confidence = np.mean([r['confidence'] for r in st.session_state.results])
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Texts", total_results)
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col3:
                st.metric("Most Common", 
                         max(['Positive', 'Negative', 'Neutral'], 
                             key=lambda x: sum(1 for r in st.session_state.results if r['primary_sentiment'] == x)))
            with col4:
                st.metric("Latest Analysis", 
                         st.session_state.results[-1]['timestamp'].strftime('%H:%M:%S'))
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = create_sentiment_distribution_chart(st.session_state.results)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_confidence = create_confidence_chart(st.session_state.results)
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Time series (if enough data)
            if len(st.session_state.results) > 1:
                fig_time = create_time_series_chart(st.session_state.results)
                if fig_time:
                    st.plotly_chart(fig_time, use_container_width=True)
            
            # Word cloud
            if len(st.session_state.results) > 0:
                st.subheader("Keywords Analysis")
                fig_wordcloud = create_word_cloud(st.session_state.results)
                if fig_wordcloud:
                    st.pyplot(fig_wordcloud)
    
    with tab4:
        st.subheader("📥 Export Your Results")
        
        if not st.session_state.results:
            st.info("🔍 **No results to export yet!**")
            st.write("**Steps to get started:**")
            st.write("1. Go to the **🔍 Text Analysis** tab")
            st.write("2. Enter some text and click **Analyze Text**")
            st.write("3. Come back here to download your results!")
            st.write("")
            st.write("**Available export formats:**")
            st.write("• **📄 CSV** - Open in Excel or Google Sheets")
            st.write("• **📋 JSON** - For developers and data processing")
            st.write("• **📑 PDF** - Professional report for sharing")
        else:
            st.success(f"✅ **{len(st.session_state.results)} analysis results ready for export!**")
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                positive_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Positive')
                st.metric("😊 Positive", positive_count)
            with col2:
                negative_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Negative')
                st.metric("😔 Negative", negative_count)
            with col3:
                neutral_count = sum(1 for r in st.session_state.results if r['primary_sentiment'] == 'Neutral')
                st.metric("😐 Neutral", neutral_count)
            with col4:
                avg_confidence = np.mean([r['confidence'] for r in st.session_state.results])
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
            
            st.write("---")
            st.subheader("📁 Download Options")
            
            # Export buttons in main content area
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📄 CSV Format**")
                st.write("Perfect for Excel analysis")
                csv_data = export_to_csv(st.session_state.results)
                st.download_button(
                    label="📄 Download CSV File",
                    data=csv_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download as spreadsheet file",
                    use_container_width=True
                )
            
            with col2:
                st.write("**📋 JSON Format**")
                st.write("Structured data format")
                json_data = export_to_json(st.session_state.results)
                st.download_button(
                    label="📋 Download JSON File",
                    data=json_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download as JSON file",
                    use_container_width=True
                )
            
            with col3:
                st.write("**📑 PDF Report**")
                st.write("Professional summary")
                try:
                    pdf_data = create_pdf_report(st.session_state.results)
                    st.download_button(
                        label="📑 Download PDF Report",
                        data=pdf_data,
                        file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download professional report",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {str(e)}")
                    st.info("💡 Try CSV or JSON export instead")
            
            st.write("---")
            st.info("💡 **Tip**: After downloading, you can continue analyzing more text. All results will be included in future exports!")
    
    with tab5:
        st.subheader("Analysis History")
        
        if not st.session_state.results:
            st.info("No analysis results yet. Start analyzing text to see history!")
        else:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_filter = st.selectbox("Filter by sentiment:", 
                                              ['All', 'Positive', 'Negative', 'Neutral'])
            
            with col2:
                min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0)
            
            with col3:
                sort_by = st.selectbox("Sort by:", ['Timestamp (newest)', 'Timestamp (oldest)', 'Confidence (high)', 'Confidence (low)'])
            
            # Apply filters
            filtered_results = st.session_state.results.copy()
            
            if sentiment_filter != 'All':
                filtered_results = [r for r in filtered_results if r['primary_sentiment'] == sentiment_filter]
            
            filtered_results = [r for r in filtered_results if r['confidence'] >= min_confidence]
            
            # Apply sorting
            if sort_by == 'Timestamp (newest)':
                filtered_results.sort(key=lambda x: x['timestamp'], reverse=True)
            elif sort_by == 'Timestamp (oldest)':
                filtered_results.sort(key=lambda x: x['timestamp'])
            elif sort_by == 'Confidence (high)':
                filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
            elif sort_by == 'Confidence (low)':
                filtered_results.sort(key=lambda x: x['confidence'])
            
            # Display results
            if filtered_results:
                for i, result in enumerate(filtered_results):
                    with st.expander(f"{result['primary_sentiment']} - {result['confidence']:.1%} - {result['timestamp'].strftime('%H:%M:%S')}"):
                        st.write("**Text:**", result['text'])
                        st.write("**Sentiment:**", result['primary_sentiment'])
                        st.write("**Confidence:**", f"{result['confidence']:.2%}")
                        if result['keywords']:
                            st.write("**Keywords:**", ', '.join(result['keywords']))
                        st.write("**Analyzed at:**", result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            else:
                st.warning("No results match your filters.")
            
            # Clear history button
            if st.button("🗑️ Clear All History", type="secondary"):
                st.session_state.results = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Built with Streamlit, Hugging Face API, and Plotly</p>
        <p>💡 Tip: Get your free Hugging Face API key at <a href='https://huggingface.co/settings/tokens' target='_blank'>huggingface.co</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()