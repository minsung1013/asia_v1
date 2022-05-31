import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import msoffcrypto

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Asia Market Analysis for PP')

passwd = st.secrets['passwd']

@st.cache(allow_output_mutation=True)
def read_excel():
    decrypted_workbook = io.BytesIO()
    with open('./data2/df_data.xlsx', 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=passwd)
        office_file.decrypt(decrypted_workbook)

    df = pd.read_excel(decrypted_workbook, sheet_name='df_data', index_col=0, engine='openpyxl')
    return df


@st.cache
def read_data_a():
    preservative = pd.read_csv('./data/df_preservative.csv', index_col=0)
    antioxidant = pd.read_csv('./data/df_antioxidant.csv', index_col=0)
    chelating = pd.read_csv('./data/df_chelating.csv', index_col=0)
    evonik = pd.read_csv('./data/df_evonik.csv', index_col=0)

    tsne = pd.read_csv('./data2/df_tsne.csv', index_col=0)
    intermediate = pd.read_pickle('./data2/df_intermediate.pkl')

    return preservative, antioxidant, chelating, evonik, tsne, intermediate

df_data = read_excel()
df_preservative, df_antioxidant, df_chelating, df_evonik, df_tsne, df_intermediate = read_data_a()

df_data.fillna('no', inplace=True)

#====================================================================================================
#====================================================================================================


st.header('select your dataset')

all = ['all']

with st.form(key='my_form1'):
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        market = st.selectbox('Market', all + list(df_data['Market'].unique()))
    with col1_2:
        market_max = st.number_input('show', min_value=1, max_value=20, value=15)

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        manufacturer = st.selectbox('Manufacturer', all + list(df_data['Manufacturer'].unique()))
    with col2_2:
        manufacturer_max = st.number_input('show', min_value=1, max_value=30, value=15)

    option_cat = st.radio('show by', ['Category', 'Class', 'Sub-Category'])
    category = st.selectbox('category', all + list(df_data['Category'].unique()))
    class_ = st.selectbox('class', all + list(df_data['Class'].unique()))
    sub_category = st.selectbox('sub-category', all + list(df_data['Sub-Category'].unique()))
    year = st.multiselect('year', all + list(df_data['Year'].unique()))

    submit_button = st.form_submit_button(label='Submit')

@st.cache
def mask_generator(string, column):
    if string != 'all':
        return df_data[column] == string
    else:
        return np.array([True] * len(df_data))

mask_market = mask_generator(market, 'Market')
mask_manufacturer = mask_generator(manufacturer, 'Manufacturer')

if option_cat == 'Category':
    mask_cat = mask_generator(category, 'Category')
elif option_cat == 'Class':
    mask_cat = mask_generator(class_, 'Class')
else:
    mask_cat = mask_generator(sub_category, 'Sub-Category')

if 'all' in year or len(year) == 0:
    mask_year = np.array([True] * len(df_data))
else:
    mask_year = np.array([False] * len(df_data))
    for y in year:
        mask_year = (mask_year) | (df_data['Year'] == y)

mask_final = (mask_market) & (mask_manufacturer) & (mask_cat) & (mask_year)

data = df_data[mask_final]
preservative = df_preservative[mask_final]
antioxidant = df_antioxidant[mask_final]
chelating = df_chelating[mask_final]
evonik = df_evonik[mask_final]

tsne = df_tsne[mask_final]
intermediate = df_intermediate[mask_final]


#====================================================================================================
#====================================================================================================

def plot_1(data, subject, size=(10,4), max=None):
    st.subheader(subject)
    fig, ax = plt.subplots(figsize=size)
    if max:
        sr_subject = data[subject].value_counts()[:max]
    else:
        sr_subject = data[subject].value_counts()

    bars = ax.bar(sr_subject.index, sr_subject, color=(153/255,29/255,133/255))
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()*(1/2),b.get_height()+1, sr_subject[i],ha='center',fontsize=13)
        ax.text(b.get_x()+b.get_width()*(1/2),b.get_height()*(1/2), round(sr_subject[i]/len(data)*100,2),
                ha='center',fontsize=10)
    plt.xticks(rotation=90)

    return fig


st.subheader('data selection')
st.write('Market :', market)
st.write('Manufacturer :', manufacturer)
st.write('Option_show :', option_cat)
# st.write('data :', ','.join(year))
st.write('num of products: ', len(data))

st.header('Overall')
with st.expander('see more'):

    st.pyplot(plot_1(data, 'Market',max=int(market_max)))
    st.pyplot(plot_1(data, option_cat))
    st.pyplot(plot_1(data, 'Manufacturer', max=int(manufacturer_max)))


#====================================================================================================
#====================================================================================================

def plot_2 (data, size=(10,4), max=10):
    st.subheader('Most frequent')
    data = data.drop('Year', axis=1)
    sr_sum = data.sum(axis=0).sort_values(ascending=False)[:max]

    fig, ax = plt.subplots(figsize=size)
    bars = ax.bar(sr_sum.index, sr_sum, color=(153/255,29/255,133/255))
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()*(1/2),b.get_height()+1, sr_sum[i],ha='center',fontsize=13)
        ax.text(b.get_x()+b.get_width()*(1/2),b.get_height()*(1/2), round(sr_sum[i]/len(data)*100,2),
                ha='center',fontsize=10)
    plt.xticks(rotation=90)
    return fig

def plot_3 (data1, data2, max=10):
    st.subheader('Trend')
    columns = data2.sum(axis=0).sort_values(ascending=False).index[1:max]
    # data2['Year'] = data1['Year']
    df_group = data2.groupby('Year')[columns].sum().T
    st.table(df_group)
    df_group = df_group/data2.groupby('Year').size()*100
    df_group.T.plot(figsize=(7,5))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


st.header('Preservative')
with st.expander('see more'):
    st.pyplot(plot_2(preservative, max=10))
    st.pyplot(plot_3(data, preservative))
    st.pyplot(plot_1(data, 'p_system', max=10))

st.header('Antioxidant')
with st.expander('see more'):
    st.pyplot(plot_2(antioxidant, max=10))
    st.pyplot(plot_3(data, antioxidant))
    st.pyplot(plot_1(data, 'a_system', max=10))

st.header('Chelating_Agent')
with st.expander('see more'):
    st.pyplot(plot_2(chelating, max=10))
    st.pyplot(plot_3(data, chelating))
    st.pyplot(plot_1(data, 'c_system', max=10))

#====================================================================================================
#====================================================================================================

st.title("Clustering model for product analysis")

select_option = ['nothing','preservative','p_system','antioxidant','a_system','chelating','c_system', 'evonik', 'e_sytem']



with st.form(key='my_form2'):
    st.subheader('product find')
    option_product = st.radio('select options', select_option)

    col1_a, col1_b = st.columns(2)
    with col1_a:
        one_p = st.selectbox('preservative', preservative.columns)
    with col1_b:
        one_p_system = st.selectbox('p_system', data['p_system'].value_counts().index)

    col2_a, col2_b = st.columns(2)
    with col2_a:
        one_a = st.selectbox('antioxidant', antioxidant.columns)
    with col2_b:
        one_a_system = st.selectbox('a_system', data['a_system'].value_counts().index)

    col3_a, col3_b = st.columns(2)
    with col3_a:
        one_c = st.selectbox('chelating', chelating.columns)
    with col3_b:
        one_c_system = st.selectbox('c_system', data['c_system'].value_counts().index)

    col4_a, col4_b = st.columns(2)
    with col4_a:
        one_e = st.selectbox('evonik', evonik.columns)
    with col4_b:
        one_e_system = st.selectbox('e_system', data['e_system'].value_counts().index)

    submit_button = st.form_submit_button(label='Submit')

def mask_generator2(string, df, column=None):
    if column:
        mask = df[column] == string
    else:
        mask = df[string] == 1
    return mask

one = [one_p, one_p_system, one_a, one_a_system, one_c, one_c_system, one_e, one_e_system]


if option_product == select_option[1]:
    mask_pf = mask_generator2(one_p, preservative)
elif option_product == select_option[2]:
    mask_pf = mask_generator2(one_p_system, data, column='p_system')
elif option_product == select_option[3]:
    mask_pf = mask_generator2(one_a, antioxidant)
elif option_product == select_option[4]:
    mask_pf = mask_generator2(one_a_system, data, column='a_system')
elif option_product == select_option[5]:
    mask_pf = mask_generator2(one_c, chelating)
elif option_product == select_option[6]:
    mask_pf = mask_generator2(one_c_system, data, column='c_system')
elif option_product == select_option[7]:
    mask_pf = mask_generator2(one_e, evonik)
elif option_product == select_option[8]:
    mask_pf = mask_generator2(one_e_system, data, column='e_system')
else:
    mask_pf = [True] * len(data)

data_pf = data[mask_pf]

tsne_pf = tsne[mask_pf]
intermediate_pf = intermediate[mask_pf]

def plot_4(data, columns, max=None):
    if max:
        key = data[columns].value_counts().index[:max]
    else:
        key = data[columns].value_counts().index

    df_group_0 = data.groupby('Year')[columns].value_counts().unstack().fillna(0).astype('int')
    df_group = data.groupby('Year')[columns].value_counts().unstack()[key].fillna(0).astype('int')
    st.table(df_group.T)
    df_group2 = round(df_group.T/df_group_0.sum(axis=1)*100, 2).T
    df_group2.fillna(0)
    df_group2.plot(figsize=(7,5))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

st.subheader('product selection')

selection = {}
for option, name in zip(select_option[1:], one):
    selection[option] = name

if option_product != 'nothing':
    st.write(option_product, " : ", selection[option_product])
st.write('number of product: ', len(data_pf))

def plot_cluster_2(tsne, df_b, key_b, tsne_sub, mask, key_f, size=10, a=0.1):
    tsne_b = pd.concat([tsne, df_b[key_b]], axis=1)
    tsne_f = tsne_sub[mask]

    eliment = tsne_b[key_b].unique()
    class_order = ['skincare_face', 'skincare_other', 'hair_product', 'colour_lip', 'cleanser', 'suncare',
                   'colour_face', 'colour_eye', 'hair_shampoo']
    tab = ['tab:blue', 'navy', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'peru',
           'tab:olive', 'tab:cyan']
    basic = ['b', 'g', 'r', 'c', 'm', 'y', 'lime', 'darkgreen', 'gold', 'navy', 'palevioletred', 'firebrick', 'peru']
    color_list = tab * 2 + basic

    key_option = ['Sub-Category', 'Category', 'Manufacturer', 'kmeans', 'Year']

    if key_b in key_option:
        dic_color = {}
        for i, e in enumerate(eliment):
            dic_color[e] = color_list[i]

    elif key_b == 'Class':
        dic_color = {}
        for i, e in enumerate(class_order):
            dic_color[e] = color_list[i]

    else:
        dic_color = {0: 'tab:blue', 1: 'tab:orange'}

    fig, ax = plt.subplots(figsize=(size, size))
    print(key_b, key_f)

    grouped = tsne_b.groupby(key_b)
    for k, group in grouped:
        group.plot(ax=ax, kind='scatter', x='0', y='1', label=k, c=dic_color[k], s=100, alpha=a)

    tsne_f.plot(ax=ax, kind='scatter', x='0', y='1', label=key_f, c='tab:orange', marker='v', s=200, alpha=1,
                edgecolor='black', linewidth=1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 20}, markerscale=2)

    return fig

st.pyplot(plot_cluster_2(df_tsne, df_data, 'Class', tsne, mask_pf, 'Target', size=25, a=0.3))


st.header('Market')
with st.expander('see more'):
    st.pyplot(plot_1(data_pf, 'Market'))
    st.pyplot(plot_4(data_pf, 'Market'))

st.header('Manufacturer')
with st.expander('see more'):
    st.pyplot(plot_1(data_pf, 'Manufacturer', max=10))
    st.pyplot(plot_4(data_pf, 'Manufacturer', max=10))

st.header('Class')
with st.expander('see more'):
    st.pyplot(plot_1(data_pf, 'Class'))
    st.pyplot(plot_4(data_pf, 'Class'))

st.header('customer report')
sub_category = data['Class'].unique()
with st.expander('see more'):
    if  manufacturer == 'all':
        st.write('please select a manufacturer')
    else:
        for sub in sub_category:
            st.text(sub)
            st.text(data_pf[data_pf['Class'] == sub]['p_system'].value_counts()[:5])
            st.write()
