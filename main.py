
import streamlit as st



from web_search import web_search
from database_search import database_search
from content_search import content_search
from document_qa import document_qa




import streamlit as st

def main():
    st.title("Multi Source ChatBot")

    # Create two columns at the top for the option selector and the query input
    col1, col2 = st.columns(2)

    with col1:
        # Left top: Option selector
        option = st.radio("Select an option", ("Web search", "DataBase Search", "Content Search", "Document QA"))

    with col2:
        # Right top: Query input box
        query = st.text_input("Enter your query:")

    # Create two columns below the first row for the file upload and content input
    col3, col4 = st.columns(2)

    with col3:
        # Left bottom: File upload option
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    with col4:
        # Right bottom: Content input area
        content = st.text_area("Enter your content:")

    # Place the process button and answer display below everything
    if st.button("Process"):
        # Call the appropriate function based on the selection
        if option == "Web search":
            answer = web_search(query)
        elif option == "DataBase Search":
            answer = database_search(query)
        elif option == "Content Search":
            answer = content_search(query, content)
        elif option == "Document QA":
            answer = document_qa(query,uploaded_file)

        # Display the answer in a text area below
        st.text_area("Answer:", value=answer)


if __name__ == "__main__":
    main()



