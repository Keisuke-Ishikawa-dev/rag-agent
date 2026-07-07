Feature: RAG Agent Question Answering
  As a user
  I want to ask questions about uploaded documents
  So that I can get accurate answers from my knowledge base

  Background:
    Given the RAG agent is running
    And the knowledge base contains documents

  Scenario: Answer question from knowledge base
    Given a document about "Python async programming" is uploaded
    When I ask "What is asyncio?"
    Then the answer should mention "event loop" or "coroutine"
    And the response should include source documents

  Scenario: Fallback to web search
    Given the knowledge base is empty
    When I ask "What is the current Bitcoin price?"
    Then the agent should use web search
    And return a relevant answer

  Scenario: Multi-turn conversation
    Given I asked "What is LangChain?"
    When I follow up with "How does it relate to RAG?"
    Then the answer should reference the previous context

  Scenario: Document upload and indexing
    Given a PDF file with 10 pages
    When I upload the document
    Then it should be split into multiple chunks
    And all chunks should be stored in the vector database
