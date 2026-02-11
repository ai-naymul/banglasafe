"""
BanglaSafe PoC Results Dashboard
================================
A Streamlit dashboard to visualize and analyze harmful prompt evaluation results
across different language scripts (English, Bengali, Banglish/Direct Bengali).

Supports both CSV and JSON result formats.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import glob as glob_module

# Page configuration
st.set_page_config(
    page_title="BanglaSafe PoC Results",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .highlight-danger {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ef5350;
    }
    .highlight-safe {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #66bb6a;
    }
    .response-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .system-msg {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .user-msg {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def find_result_files(results_dir: str) -> dict:
    """Find all result files (CSV and JSON) in the results directory."""
    files = {"csv": [], "json": []}

    # Look for CSV files
    csv_pattern = os.path.join(results_dir, "*.csv")
    files["csv"] = sorted(glob_module.glob(csv_pattern), key=os.path.getmtime, reverse=True)

    # Look for JSON files
    json_pattern = os.path.join(results_dir, "*.json")
    files["json"] = sorted(glob_module.glob(json_pattern), key=os.path.getmtime, reverse=True)

    return files


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess CSV results."""
    df = pd.read_csv(file_path)

    # Handle old format columns
    if 'auto_refused' in df.columns:
        df['auto_refused'] = df['auto_refused'].fillna('NO').str.upper().str.strip()
        df['refused'] = df['auto_refused'] == 'YES'

    if 'script' in df.columns:
        df['script'] = df['script'].str.lower().str.strip()
        # Rename for consistency
        df['prompt_type'] = df['script']

    if 'model_short' not in df.columns and 'model' in df.columns:
        df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1] if '/' in str(x) else x)

    if 'category' in df.columns:
        df['category'] = df['category'].str.replace('_', ' ').str.title()

    # Add missing columns
    if 'system_message' not in df.columns:
        df['system_message'] = None
    if 'user_message' not in df.columns:
        df['user_message'] = df.get('prompt_text', '')
    if 'language' not in df.columns:
        df['language'] = df['prompt_type'].apply(
            lambda x: 'bengali' if 'bengali' in str(x).lower() or 'banglish' in str(x).lower() else 'english'
        )

    return df


def load_json_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess JSON results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Normalize column names
    if 'model_name' in df.columns:
        df['model_short'] = df['model_name']

    if 'category' in df.columns:
        df['category'] = df['category'].str.replace('_', ' ').str.title()

    # Ensure prompt_type exists
    if 'prompt_type' not in df.columns and 'script' in df.columns:
        df['prompt_type'] = df['script']

    # Handle missing columns
    if 'prompt_text' not in df.columns:
        df['prompt_text'] = df.get('user_message', '')

    return df


@st.cache_data
def load_data(file_path: str = None) -> pd.DataFrame:
    """Load data from file or find the most recent results."""

    if file_path:
        if file_path.endswith('.json'):
            return load_json_data(file_path)
        else:
            return load_csv_data(file_path)

    # Try to find results directory
    possible_dirs = [
        "../results",
        "results",
        os.path.join(os.path.dirname(__file__), "..", "results"),
        "/Users/escobarsmacbook/Workspace/bengalisafe/evaluate_harmful_prompts/results"
    ]

    results_dir = None
    for d in possible_dirs:
        if os.path.isdir(d):
            results_dir = d
            break

    if not results_dir:
        st.error("Could not find results directory.")
        st.stop()

    # Find result files
    files = find_result_files(results_dir)

    # Prefer JSON over CSV (newer format)
    if files["json"]:
        return load_json_data(files["json"][0])
    elif files["csv"]:
        return load_csv_data(files["csv"][0])
    else:
        st.error("No result files found in the results folder.")
        st.stop()


def create_refusal_heatmap(df: pd.DataFrame):
    """Create a heatmap showing refusal rates by model and prompt type."""
    # Use prompt_type for grouping
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'
    model_col = 'model_short' if 'model_short' in df.columns else 'model_name'

    pivot = df.groupby([model_col, group_col])['refused'].mean().unstack(fill_value=0) * 100

    # Reorder columns if possible
    col_order = ['english', 'bengali', 'banglish', 'direct_bengali']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig = px.imshow(
        pivot,
        labels=dict(x="Prompt Type", y="Model", color="Refusal Rate (%)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=".1f"
    )

    fig.update_layout(
        title="Refusal Rate Heatmap: Models vs Prompt Types",
        xaxis_title="Prompt Type",
        yaxis_title="Model",
        height=400
    )

    return fig


def create_refusal_comparison_chart(df: pd.DataFrame):
    """Create a grouped bar chart comparing refusal rates."""
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'
    model_col = 'model_short' if 'model_short' in df.columns else 'model_name'

    refusal_by_type = df.groupby([model_col, group_col])['refused'].mean().reset_index()
    refusal_by_type['refused'] = refusal_by_type['refused'] * 100

    color_map = {
        'english': '#4CAF50',
        'bengali': '#FF9800',
        'banglish': '#F44336',
        'direct_bengali': '#9C27B0'
    }

    fig = px.bar(
        refusal_by_type,
        x=model_col,
        y='refused',
        color=group_col,
        barmode='group',
        labels={'refused': 'Refusal Rate (%)', model_col: 'Model', group_col: 'Prompt Type'},
        title='Refusal Rate Comparison by Model and Prompt Type',
        color_discrete_map=color_map
    )

    fig.update_layout(height=450, xaxis_tickangle=-45)

    return fig


def create_category_breakdown(df: pd.DataFrame):
    """Create a breakdown by harm category."""
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'

    category_data = df.groupby(['category', group_col])['refused'].mean().reset_index()
    category_data['refused'] = category_data['refused'] * 100

    color_map = {
        'english': '#4CAF50',
        'bengali': '#FF9800',
        'banglish': '#F44336',
        'direct_bengali': '#9C27B0'
    }

    fig = px.bar(
        category_data,
        x='category',
        y='refused',
        color=group_col,
        barmode='group',
        labels={'refused': 'Refusal Rate (%)', 'category': 'Harm Category', group_col: 'Prompt Type'},
        title='Refusal Rate by Harm Category and Prompt Type',
        color_discrete_map=color_map
    )

    fig.update_layout(height=400, xaxis_tickangle=-45)

    return fig


def create_safety_gap_chart(df: pd.DataFrame):
    """Create a chart showing the safety gap between English and other prompt types."""
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'
    model_col = 'model_short' if 'model_short' in df.columns else 'model_name'

    eng_refusal = df[df[group_col] == 'english'].groupby(model_col)['refused'].mean()

    # Get other language refusal rates
    other_types = [t for t in df[group_col].unique() if t != 'english']

    gap_data = pd.DataFrame({'model': eng_refusal.index})
    gap_data['English Refusal'] = eng_refusal.values * 100

    for ptype in other_types:
        type_refusal = df[df[group_col] == ptype].groupby(model_col)['refused'].mean()
        # Align indices
        gap_values = []
        for model in gap_data['model']:
            if model in type_refusal.index and model in eng_refusal.index:
                gap_values.append((eng_refusal[model] - type_refusal[model]) * 100)
            else:
                gap_values.append(0)
        gap_data[f'{ptype.title()} Gap'] = gap_values

    fig = go.Figure()

    colors = {'Bengali Gap': '#FF9800', 'Banglish Gap': '#F44336', 'Direct_Bengali Gap': '#9C27B0'}

    for col in gap_data.columns:
        if 'Gap' in col:
            color = colors.get(col, '#607D8B')
            fig.add_trace(go.Bar(
                name=col.replace('_', ' '),
                x=gap_data['model'],
                y=gap_data[col],
                marker_color=color
            ))

    fig.update_layout(
        title='Safety Gap: English Refusal Rate - Other Prompt Type Refusal Rate',
        xaxis_title='Model',
        yaxis_title='Safety Gap (percentage points)',
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def create_response_length_analysis(df: pd.DataFrame):
    """Analyze response lengths by prompt type and refusal status."""
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'

    fig = px.box(
        df,
        x=group_col,
        y='response_length',
        color='refused',
        labels={'response_length': 'Response Length (chars)', group_col: 'Prompt Type', 'refused': 'Refused'},
        title='Response Length Distribution by Prompt Type and Refusal Status',
        color_discrete_map={True: '#4CAF50', False: '#F44336'}
    )

    fig.update_layout(height=400)

    return fig


def display_prompt_response(row: pd.Series, show_system: bool = True):
    """Display a prompt-response pair in a nice format."""

    refused = row.get('refused', False)
    status_color = "green" if refused else "red"
    status_text = "REFUSED" if refused else "COMPLIED"

    prompt_type = row.get('prompt_type', row.get('script', 'unknown'))
    st.markdown(f"### {prompt_type.replace('_', ' ').title()} :{status_color}[{status_text}]")

    # Show system message if present
    if show_system and pd.notna(row.get('system_message')) and row.get('system_message'):
        st.markdown("**System Message:**")
        st.markdown(f'<div class="system-msg">{row["system_message"]}</div>', unsafe_allow_html=True)

    # Show user message
    user_msg = row.get('user_message', row.get('prompt_text', ''))
    st.markdown("**User Message:**")
    st.markdown(f'<div class="user-msg">{user_msg}</div>', unsafe_allow_html=True)

    # Show response
    st.markdown("**Model Response:**")
    response = str(row.get('response', ''))

    if refused:
        st.success(response[:2000] + ("..." if len(response) > 2000 else ""))
    else:
        st.error(response[:2000] + ("..." if len(response) > 2000 else ""))

    # Show metadata
    with st.expander("Details"):
        cols = st.columns(4)
        cols[0].write(f"**Length:** {row.get('response_length', len(response))}")
        cols[1].write(f"**Language:** {row.get('language', 'N/A')}")
        cols[2].write(f"**Confidence:** {row.get('refusal_confidence', 'N/A')}")
        cols[3].write(f"**Error:** {row.get('error', 'None')}")


def main():
    # Header
    st.markdown('<div class="main-header">BanglaSafe PoC Results Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyzing Safety Alignment Gaps in Bengali Language Models</div>', unsafe_allow_html=True)

    # Sidebar - File selection
    st.sidebar.header("Data Source")

    # Find available result files
    possible_dirs = [
        "../results",
        "results",
        os.path.join(os.path.dirname(__file__), "..", "results")
    ]

    results_dir = None
    for d in possible_dirs:
        if os.path.isdir(d):
            results_dir = d
            break

    selected_file = None
    if results_dir:
        files = find_result_files(results_dir)
        all_files = files["json"] + files["csv"]

        if all_files:
            file_names = [os.path.basename(f) for f in all_files]
            selected_idx = st.sidebar.selectbox(
                "Select Result File",
                range(len(file_names)),
                format_func=lambda x: file_names[x]
            )
            selected_file = all_files[selected_idx]

    # Load data
    df = load_data(selected_file)

    # Get column names for filtering
    group_col = 'prompt_type' if 'prompt_type' in df.columns else 'script'
    model_col = 'model_short' if 'model_short' in df.columns else 'model_name'

    # Sidebar filters
    st.sidebar.header("Filters")

    # Model filter
    models = ['All'] + sorted(df[model_col].unique().tolist())
    selected_model = st.sidebar.selectbox("Select Model", models)

    # Category filter
    categories = ['All'] + sorted(df['category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)

    # Prompt type filter
    prompt_types = ['All'] + sorted(df[group_col].unique().tolist())
    selected_prompt_type = st.sidebar.selectbox("Select Prompt Type", prompt_types)

    # Apply filters
    filtered_df = df.copy()
    if selected_model != 'All':
        filtered_df = filtered_df[filtered_df[model_col] == selected_model]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_prompt_type != 'All':
        filtered_df = filtered_df[filtered_df[group_col] == selected_prompt_type]

    # Key Metrics Row
    st.header("Key Findings")

    # Calculate metrics
    prompt_types_list = df[group_col].unique().tolist()
    has_english = 'english' in prompt_types_list
    has_bengali = 'bengali' in prompt_types_list
    has_direct = 'direct_bengali' in prompt_types_list

    num_metrics = 2 + sum([has_english, has_bengali, has_direct])
    cols = st.columns(min(num_metrics, 5))

    col_idx = 0
    with cols[col_idx]:
        total_prompts = len(filtered_df)
        st.metric("Total Evaluations", f"{total_prompts:,}")
    col_idx += 1

    with cols[col_idx]:
        overall_refusal = filtered_df['refused'].mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Overall Refusal Rate", f"{overall_refusal:.1f}%")
    col_idx += 1

    eng_refusal = 0
    if has_english:
        with cols[col_idx]:
            eng_data = df[df[group_col] == 'english']
            eng_refusal = eng_data['refused'].mean() * 100 if len(eng_data) > 0 else 0
            st.metric("English Refusal", f"{eng_refusal:.1f}%",
                      help="Baseline refusal rate for English prompts")
        col_idx += 1

    if has_bengali:
        with cols[col_idx]:
            ben_data = df[df[group_col] == 'bengali']
            ben_refusal = ben_data['refused'].mean() * 100 if len(ben_data) > 0 else 0
            delta = ben_refusal - eng_refusal if has_english else 0
            st.metric("Bengali Refusal", f"{ben_refusal:.1f}%",
                      delta=f"{delta:+.1f}%" if has_english else None,
                      delta_color="inverse")
        col_idx += 1

    if has_direct and col_idx < len(cols):
        with cols[col_idx]:
            direct_data = df[df[group_col] == 'direct_bengali']
            direct_refusal = direct_data['refused'].mean() * 100 if len(direct_data) > 0 else 0
            delta = direct_refusal - eng_refusal if has_english else 0
            st.metric("Direct Bengali", f"{direct_refusal:.1f}%",
                      delta=f"{delta:+.1f}%" if has_english else None,
                      delta_color="inverse")

    # Safety Gap Alert
    if has_english:
        other_refusals = []
        for pt in prompt_types_list:
            if pt != 'english':
                pt_data = df[df[group_col] == pt]
                if len(pt_data) > 0:
                    other_refusals.append(pt_data['refused'].mean() * 100)

        if other_refusals:
            min_other = min(other_refusals)
            safety_gap = eng_refusal - min_other
            if safety_gap > 10:
                st.warning(f"**Critical Safety Gap Detected:** {safety_gap:.1f} percentage point difference between English and lowest non-English refusal rate!")

    st.divider()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Model Analysis",
        "Category Breakdown",
        "Prompt Explorer",
        "Raw Data"
    ])

    with tab1:
        st.subheader("Safety Alignment Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_refusal_heatmap(df), use_container_width=True)

        with col2:
            if has_english:
                st.plotly_chart(create_safety_gap_chart(df), use_container_width=True)
            else:
                st.plotly_chart(create_refusal_comparison_chart(df), use_container_width=True)

        # Summary statistics table
        st.subheader("Summary Statistics by Prompt Type")

        summary = df.groupby(group_col).agg({
            'refused': ['count', 'sum', 'mean'],
            'response_length': 'mean'
        }).round(2)

        summary.columns = ['Total Prompts', 'Refused Count', 'Refusal Rate', 'Avg Response Length']
        summary['Refusal Rate'] = (summary['Refusal Rate'] * 100).round(1).astype(str) + '%'
        summary['Avg Response Length'] = summary['Avg Response Length'].round(0).astype(int)

        st.dataframe(summary, use_container_width=True)

    with tab2:
        st.subheader("Model-by-Model Analysis")

        st.plotly_chart(create_refusal_comparison_chart(df), use_container_width=True)

        # Model-specific details
        st.subheader("Detailed Model Statistics")

        model_stats = df.groupby([model_col, group_col]).agg({
            'refused': ['count', 'sum', 'mean']
        }).round(3)

        model_stats.columns = ['Prompts', 'Refused', 'Refusal Rate']
        model_stats['Refusal Rate'] = (model_stats['Refusal Rate'] * 100).round(1).astype(str) + '%'
        model_stats = model_stats.reset_index()

        st.dataframe(model_stats, use_container_width=True, height=400)

        # Response length analysis
        st.subheader("Response Length Analysis")
        st.plotly_chart(create_response_length_analysis(df), use_container_width=True)

    with tab3:
        st.subheader("Harm Category Analysis")

        st.plotly_chart(create_category_breakdown(df), use_container_width=True)

        # Category-specific stats
        st.subheader("Category Statistics")

        cat_stats = df.groupby(['category', group_col]).agg({
            'refused': ['count', 'mean']
        }).round(3)

        cat_stats.columns = ['Prompts', 'Refusal Rate']
        cat_stats['Refusal Rate'] = (cat_stats['Refusal Rate'] * 100).round(1).astype(str) + '%'
        cat_stats = cat_stats.reset_index()

        st.dataframe(cat_stats, use_container_width=True, height=400)

        # Most problematic categories
        st.subheader("Most Problematic Categories (Lowest Refusal Rates)")

        non_english = df[df[group_col] != 'english'] if has_english else df
        if len(non_english) > 0:
            problem_cats = non_english.groupby('category')['refused'].mean().sort_values()

            for cat, rate in problem_cats.head(5).items():
                if has_english:
                    eng_rate = df[(df['category'] == cat) & (df[group_col] == 'english')]['refused'].mean()
                    gap = (eng_rate - rate) * 100
                    st.write(f"**{cat}**: {rate*100:.1f}% refusal (Gap from English: {gap:.1f}%)")
                else:
                    st.write(f"**{cat}**: {rate*100:.1f}% refusal")

    with tab4:
        st.subheader("Prompt & Response Explorer")

        # Filter options for explorer
        col1, col2, col3 = st.columns(3)

        with col1:
            explorer_model = st.selectbox("Model", df[model_col].unique(), key='explorer_model')

        with col2:
            explorer_category = st.selectbox("Category", df['category'].unique(), key='explorer_category')

        with col3:
            show_refused = st.radio("Show", ["All", "Refused Only", "Complied Only"], horizontal=True)

        # Filter data
        explorer_df = df[
            (df[model_col] == explorer_model) &
            (df['category'] == explorer_category)
        ]

        if show_refused == "Refused Only":
            explorer_df = explorer_df[explorer_df['refused'] == True]
        elif show_refused == "Complied Only":
            explorer_df = explorer_df[explorer_df['refused'] == False]

        # Get unique prompt IDs
        prompt_ids = explorer_df['prompt_id'].unique()

        if len(prompt_ids) == 0:
            st.info("No prompts match the current filters.")
        else:
            selected_prompt_id = st.selectbox("Select Prompt ID", prompt_ids)

            prompt_data = explorer_df[explorer_df['prompt_id'] == selected_prompt_id]

            st.divider()

            # Show prompts and responses for each prompt type
            prompt_types_order = ['english', 'bengali', 'banglish', 'direct_bengali']
            available_types = [pt for pt in prompt_types_order if pt in prompt_data[group_col].values]

            # Add any other types not in the standard order
            for pt in prompt_data[group_col].unique():
                if pt not in available_types:
                    available_types.append(pt)

            for ptype in available_types:
                type_data = prompt_data[prompt_data[group_col] == ptype]

                if len(type_data) > 0:
                    row = type_data.iloc[0]
                    display_prompt_response(row, show_system=True)
                    st.divider()

    with tab5:
        st.subheader("Raw Data")

        # Column selector
        all_columns = filtered_df.columns.tolist()
        default_columns = [c for c in [model_col, 'prompt_id', 'category', group_col, 'refused', 'response_length', 'refusal_confidence'] if c in all_columns]
        selected_columns = st.multiselect("Select columns to display", all_columns, default=default_columns)

        if selected_columns:
            st.dataframe(
                filtered_df[selected_columns],
                use_container_width=True,
                height=600
            )

        # Download options
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download as CSV",
                filtered_df.to_csv(index=False).encode('utf-8'),
                "filtered_results.csv",
                "text/csv"
            )

        with col2:
            st.download_button(
                "Download as JSON",
                filtered_df.to_json(orient='records', force_ascii=False, indent=2),
                "filtered_results.json",
                "application/json"
            )

    # Footer
    st.divider()
    st.markdown("""
    ---
    **BanglaSafe PoC** | Demonstrating safety alignment gaps in Bengali language processing
    *This dashboard visualizes results from testing harmful prompts across English, Bengali, and Direct Bengali scripts.*
    """)


if __name__ == "__main__":
    main()
