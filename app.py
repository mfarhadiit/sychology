import os
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 🔐 کلید API
os.environ["OPENAI_API_KEY"] = "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"


# 💬 مدل زبانی و حافظه‌ها
llm = ChatOpenAI(model="gpt-4o-ئهده", temperature=0.7)

short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([""], embedding)
retriever = vectorstore.as_retriever()

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=short_term_memory
)

# 🌐 ایجاد اپ Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "روانشناس هوشمند"

app.layout = dbc.Container([
    html.H2("💬 چت با روانشناس هوشمند", className="my-3 text-center"),
    dbc.Card([
        dbc.CardBody([
            dcc.Textarea(
                id='chat-history',
                value="",
                style={'width': '100%', 'height': 300, 'backgroundColor': '#111', 'color': '#fff'},
                readOnly=True,
            ),
            dbc.InputGroup([
                dbc.Input(id="user-input", placeholder="پیام خود را وارد کنید...", type="text"),
                dbc.Button("ارسال", id="send-button", n_clicks=0, color="primary")
            ], className="mt-3"),
            dcc.Store(id="memory-store", data=[])
        ])
    ])
], fluid=True)

@app.callback(
    Output('chat-history', 'value'),
    Output('memory-store', 'data'),
    Input('send-button', 'n_clicks'),
    State('user-input', 'value'),
    State('chat-history', 'value'),
    State('memory-store', 'data'),
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history, memory_store):
    if not user_input.strip():
        return chat_history, memory_store

    # 🔁 پرسش به زنجیره LangChain
    result = chain({"question": user_input})
    answer = result["answer"]

    # ✍️ افزودن به حافظه بلندمدت
    vectorstore.add_texts([user_input, answer])

    # 📝 به‌روزرسانی نمایش چت
    new_chat = chat_history + f"\n🧑: {user_input}\n🤖: {answer}\n"
    memory_store.append({"user": user_input, "bot": answer})

    return new_chat, memory_store

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=1000)
