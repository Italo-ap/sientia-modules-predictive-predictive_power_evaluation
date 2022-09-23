# native imports

# common imports
import numpy as np
import pandas as pd
from PIL import Image
image1 = Image.open('images/sppe_workflow_overview.png')
image2 = Image.open('images/aiglogo.png')
# special imports

# front-end imports
import streamlit as st



st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="Sientia - SPPE Module",
    page_icon='🏠',  # String, anything supported by st.image, or None.
)

def main():
    
    """Main function of the App"""

    with st.sidebar:
        #st.markdown("[![Aignosi](https://github.com/Aignosi/sientia-predictivepower-evaluation/blob/dev-refactoring/images/aiglogo.png)](http://aignosi.com)")
       
        st.sidebar.title("Navigation")
        
         #st.caption("Check out this [article](https://www.linkedin.com/pulse/visualizing-non-linear-career-path-data-and-stories/?trackingId=7RX%2Bbl6LRu2jaa5BB4aC4Q%3D%3D) to learn more about the non-linear career path.")

        st.sidebar.header("About")
        st.sidebar.info(
            """
            Part of this app is maintained by Aignosi. You can learn more about us at
            [Aignosi.com](https://aignosi.com)
            """
        )
        st.sidebar.header("Contact")
        st.sidebar.info("This is a demo version of the Sientia Predictive Power Evaluator (SPPE). For more information, contact us by email: alexandre@aignosi.com")


    # ===========================================
    # Body
    # ===========================================
    
    #st.title('Qual problema esse app resolve?')
    st.markdown('')
    st.markdown('')

    st.title('Welcome to Sientia Predictive Power Evaluator (SPPE) Module!')
    #st.image(image2,use_column_width=True)
    
    st.markdown('')
    st.subheader("The app that assesses the predictive potential of your industrial process in minutes")
    st.markdown(' ')
    st.markdown(' ')
    #st.image(image, caption='Arquitetura',use_column_width=True)
    st.subheader('**Why SPPE?**')
    st.markdown('If you need to predict a process or quality variable and you do nott know if it iss feasible to make a predictive model for it, then Pandora Box is the right app to answer that question.')
    st.markdown('')   
    st.subheader('**What are the benefits of SPPE?**')
    st.markdown('''
    
    >1. Quick analysis of the predictive potential of your problem, in a simple, fast and uncomplicated way;

    >2. Analysis of the time delay between variables, allowing to evaluate the synchronism of the data in time;

    >3. Correlation calculations that take into account the non-linear dynamics of the process
    
    >4. Causality and statistical significance tests (coming soon!).

    ''')
       
    st.markdown('')

    st.subheader('**How does it work?**')
    st.markdown("Just load a small database in *.CSV format and navigate in the application that evaluates that SPPE is already calculating everything automatically. In the end, the app **estimates the predictive potential of your problem**, allowing a more thorough assessment of the feasibility of predictively modeling the problem.")
    st.image(image1,use_column_width=True)#, caption='Arquitetura')
    st.markdown('')
    
    st.markdown('')
    st.subheader("**Let's try it?**")
    st.markdown("In the navigation bar on the left, select **Data Preparation**")
    st.markdown('')
    st.markdown('')


if __name__ == "__main__":
    main()
