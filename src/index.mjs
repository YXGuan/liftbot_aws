import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "langchain/schema/output_parser";

// import { retriever } from "/utils/retriever";

import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { createClient } from "@supabase/supabase-js";

const openAIApiKey = process.env.OPENAI_API_KEY;
const embeddings = new OpenAIEmbeddings({ openAIApiKey });
const sbApiKey = process.env.SUPABASE_API_KEY;
const sbUrl = process.env.SUPABASE_URL_LC_CHATBOT;
const client = createClient(sbUrl, sbApiKey);
const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: "liftdocsshortlong1",
  queryName: "match_liftdocs_shortlong1",
});
// const vectorStore = new SupabaseVectorStore(embeddings, {
//     client,
//     tableName: "documents",
//     queryName: "match_documents",
//   });
const retriever = vectorStore.asRetriever();

// import { Document } from "@langchain/core/documents";

// import { combineDocuments } from "/utils/combineDocuments";
function combineDocuments(docs) {
  return docs
    .map(function (doc) {
      return `Content: ${doc.content}\n Source: ${doc.metadata} \n`;
    })
    .join("\n\n");
}

import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";

// import { formatConvHistory } from "/utils/formatConvHistory";
function formatConvHistory(messages) {
  return messages
    .map((message, i) => {
      if (i % 2 === 0) {
        return `Human: ${message}`;
      } else {
        return `AI: ${message}`;
      }
    })
    .join("\n");
}

const chatbotConversation = document.getElementById(
  "chatbot-conversation-container",
);
const newAiSpeechBubble = document.createElement("div");
newAiSpeechBubble.classList.add("speech", "speech-ai");
chatbotConversation.appendChild(newAiSpeechBubble);
newAiSpeechBubble.textContent =
  "Bonjour, my name is Luke, and I am a Lift Church chatbot. You can ask me questions about the church. How may I help today?";
chatbotConversation.scrollTop = chatbotConversation.scrollHeight;

document.addEventListener("submit", (e) => {
  e.preventDefault();
  progressConversation();
});

// const openAIApiKey = process.env.OPENAI_API_KEY;
const llm = new ChatOpenAI({ openAIApiKey });

const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
conversation history: {conv_history}
question: {question} 
standalone question:`;
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate,
);

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Lift Church based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. Always try your best to answer the question. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email info@liftchurch.ca Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
conversation history: {conv_history}
question: {question}
answer: `;
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const standaloneQuestionChain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]);

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

const chain = RunnableSequence.from([
  {
    standalone_question: standaloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({ original_input }) => original_input.question,
    conv_history: ({ original_input }) => original_input.conv_history,
  },
  answerChain,
]);

const convHistory = [];

async function progressConversation() {
  const userInput = document.getElementById("user-input");
  const chatbotConversation = document.getElementById(
    "chatbot-conversation-container",
  );
  const question = userInput.value;
  userInput.value = "";

  // add human message
  const newHumanSpeechBubble = document.createElement("div");
  newHumanSpeechBubble.classList.add("speech", "speech-human");
  chatbotConversation.appendChild(newHumanSpeechBubble);
  newHumanSpeechBubble.textContent = question;
  chatbotConversation.scrollTop = chatbotConversation.scrollHeight;
  const response = await chain.invoke({
    question: question,
    conv_history: formatConvHistory(convHistory),
  });
  convHistory.push(question);
  convHistory.push(response);

  const { error } = await client
    .from("liftchathistory")
    .insert({ qora: "r", chathistory: response });

  const { error2 } = await client
    .from("liftchathistory")
    .insert({ qora: "q", chathistory: question });
  // add AI message
  const newAiSpeechBubble = document.createElement("div");
  newAiSpeechBubble.classList.add("speech", "speech-ai");
  chatbotConversation.appendChild(newAiSpeechBubble);
  newAiSpeechBubble.textContent = response;
  chatbotConversation.scrollTop = chatbotConversation.scrollHeight;
}

// import "./styles.css";

// document.getElementById("app").innerHTML = `
// <h1>Hello JavaScript!</h1>
// `;
