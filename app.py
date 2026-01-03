import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG-Based Chatbot")

# ----------------------
# DOCUMENT DATA (INLINE)
# ----------------------
DOCUMENT_TEXT = """
Parallel computing is the simultaneous use of multiple compute resources to solve a computational problem. It involves dividing a large problem into smaller subproblems that can be solved concurrently, thereby reducing the total execution time. The key idea is to exploit concurrency to improve speed. There are two common types of parallelism: data parallelism, where the same operation is performed on different parts of the data, and task parallelism, where different tasks are executed in parallel.
Distributed computing involves multiple autonomous computers connected via a network working together to achieve a common goal. Each computer has its own memory and processor, and coordination happens through message passing or shared communication. Distributed systems are characterized by scalability, fault tolerance, and resource sharing.
The importance of parallel and distributed computing lies in its ability to handle modern problems that require massive computational power, such as climate modeling, big data analytics, and artificial intelligence. It enables handling larger datasets and more complex simulations more cost-effectively than single supercomputers.
There are various models of parallel and distributed computation. Parallel computation models include PRAM (Parallel Random Access Machine), an abstract model with unlimited processors sharing a common memory; Bulk Synchronous Parallel (BSP), in which computation proceeds in supersteps separated by barrier synchronization; and dataflow models, where execution is driven by data availability.
Distributed computation models include the message passing model, where processes communicate by sending and receiving messages; the shared memory model, which abstracts a distributed shared memory though physically distributed; and the actor model, where independent actors communicate asynchronously via message passing.
Flynn’s taxonomy classifies computer architectures based on instruction and data streams: SISD (single instruction, single data) for traditional sequential CPUs; SIMD (single instruction, multiple data) for vector processors and GPUs; MISD (multiple instruction, single data), which is rare and used in fault-tolerant systems; and MIMD (multiple instruction, multiple data) for multiprocessors and clusters.
Shared memory architectures have multiple processors sharing a global memory space, which simplifies programming but introduces challenges such as synchronization and maintaining memory consistency. Distributed memory architectures assign each processor its own local memory, requiring communication via message passing but allowing scalability to large systems. Hybrid architectures combine features of both shared and distributed memory, such as clusters of multicore nodes.
Distributed systems consist of nodes (computers), network infrastructure, and middleware for communication and coordination. Communication models in distributed systems can be synchronous or asynchronous and include mechanisms like Remote Procedure Call (RPC) and message queues or brokers. Resource management involves scheduling, load balancing, and fault detection and recovery.
Designing parallel algorithms involves decomposition (dividing the problem into tasks), mapping (assigning tasks to processors), communication (designing efficient data exchange), and synchronization (coordinating task execution). Common parallel algorithms include sorting algorithms like parallel merge sort and bitonic sort; matrix multiplication using block-based techniques; graph algorithms such as breadth-first search (BFS) and shortest path calculations; and numerical methods like Jacobi iteration.
Programming models for parallel computing include shared memory programming with threads (e.g., POSIX Threads, OpenMP) using synchronization primitives like mutexes and semaphores; message passing programming using MPI (Message Passing Interface) with explicit send/receive calls; data parallel languages such as CUDA for GPUs and OpenCL for heterogeneous computing; and distributed programming frameworks like MapReduce (Hadoop) and Spark for in-memory distributed computing.
Communication in parallel and distributed systems occurs in two main types: point-to-point communication between two processes and collective communication involving groups of processes (such as broadcast, scatter, or gather operations). Communication costs include latency (time to start communication), bandwidth (data transfer rate), and overheads from synchronization and contention.
Synchronization mechanisms are essential to coordinate parallel tasks. Barriers make all processes wait until each has reached a certain point. Locks and mutexes provide mutual exclusion to protect shared data access. Condition variables and semaphores allow signaling between threads or processes.
Performance metrics include speedup, defined as the ratio of serial execution time to parallel execution time: S=T_serial/T_parallel . Efficiency measures how well processors are utilized: E=S/p, where pis the number of processors. Scalability refers to the system’s ability to maintain efficiency as the problem size or number of processors increases. Amdahl’s Law limits maximum speedup by the fraction of code that must be executed serially: S_max=1/((1-f)+f/p), where fis the parallelizable fraction.
Fault tolerance is critical in distributed systems because failures are inevitable. Types of failures include crash failures, Byzantine failures (arbitrary faults), and network partitions. Fault detection techniques involve heartbeat messages and watchdog timers. Recovery methods include checkpointing (saving system state periodically) and replication of data or processes.
Applications of parallel and distributed computing span scientific simulations like weather prediction and molecular dynamics; big data analytics processing huge datasets using distributed frameworks; machine learning and AI training large models using GPUs or clusters; and cloud computing services that share resources over the Internet.
Future trends include integrating quantum computing techniques with classical parallel systems, edge computing which brings computation closer to data sources to reduce latency, improved energy-efficient architectures, and AI-driven optimization accelerating the development of parallel programs.
This comprehensive overview covers fundamental concepts, models, architectures, algorithms, programming methods, communication, synchronization, performance evaluation, fault tolerance, applications, and future directions of parallel distributed computing. If you would like more detailed explanations or examples on any particular aspect, please let me know.
Parallel and distributed computing has evolved over decades, driven by the constant need for faster, more efficient processing to handle increasingly complex problems and large datasets. The core objective is to divide a computational task into smaller parts that can be processed simultaneously, either within a single machine (parallel computing) or across multiple machines (distributed computing).
In parallel computing, multiple processors or cores work collaboratively by sharing memory or communicating through high-speed interconnections to execute parts of a program concurrently. This approach reduces the overall runtime significantly compared to a sequential execution, especially for compute-intensive tasks.
Distributed computing, on the other hand, involves a network of independent computers working together. Each node in the network operates autonomously with its own local memory and communication takes place via message passing over the network. Distributed systems can be geographically dispersed and are designed to be scalable and fault tolerant.
One important aspect of parallel and distributed computing is understanding the architectural models that determine how processors and memory are organized and how they communicate:
	Shared Memory Architectures: Multiple processors access a common physical memory. Systems like symmetric multiprocessors (SMP) fall under this category. They offer ease of programming but face issues like contention and consistency.
	Distributed Memory Architectures: Each processor has its own private memory, and processors communicate via message passing. This model is typical in clusters and massively parallel processors (MPPs). It scales better but requires explicit communication management.
	Hybrid Architectures: Combine both shared and distributed memory features for improved scalability and ease of programming. Examples include clusters of multicore processors.
Programming in parallel and distributed environments requires specialized models and tools:
	Thread-based programming (e.g., OpenMP) is common in shared memory systems, allowing developers to create threads that run concurrently while managing synchronization with locks, barriers, and atomic operations.
	Message Passing Interface (MPI) is a standard for programming distributed memory systems where processes communicate explicitly by sending and receiving messages.
	GPU programming platforms like CUDA and OpenCL enable data-parallel computations on thousands of cores optimized for vectorized operations.
	High-level frameworks such as MapReduce and Apache Spark simplify distributed data processing by abstracting the underlying complexity of parallelism and fault tolerance.
Performance analysis in parallel computing involves measuring speedup, efficiency, scalability, and identifying bottlenecks such as communication overhead or load imbalance. Amdahl’s Law provides theoretical limits on speedup achievable based on the fraction of serial code, while Gustafson’s Law argues that scaling problem size can lead to better utilization of parallel resources.
Synchronization techniques are crucial to ensure correctness when multiple processes or threads access shared resources. Locks prevent simultaneous modifications but can cause deadlocks if not managed carefully. Barrier synchronization forces threads or processes to wait until all have reached a certain point, ensuring coordinated progress.
Distributed systems add complexity with challenges such as network latency, partial failures, inconsistent states, and security concerns. Fault tolerance mechanisms like checkpointing save system state periodically so computations can resume from the last checkpoint after failures. Replication improves availability by maintaining copies of data or services across different nodes.
Applications of parallel and distributed computing are widespread: scientific simulations model phenomena in physics or biology requiring intensive computation; big data analytics process massive datasets for insights; machine learning leverages parallelism to train complex models faster; cloud computing delivers on-demand resources globally; blockchain technology uses distributed consensus mechanisms for secure transactions.
Looking ahead, trends include heterogeneous computing where CPUs, GPUs, FPGAs, and AI accelerators work together; edge computing bringing computation closer to data sources; advancements in quantum computing offering new paradigms; and more intelligent compilers and runtime environments that automatically optimize parallel execution.
Expanding on each aspect:
Decomposition Techniques: To efficiently parallelize a problem, it is necessary to divide it into smaller tasks. Decomposition can be task-based where different tasks run concurrently or data-based where the same operation applies to different chunks of data.
Load Balancing: Distributing tasks evenly among processors is vital to avoid some processors being idle while others are overloaded. Static load balancing assigns tasks at compile time whereas dynamic load balancing redistributes work during execution.
Communication Overheads: In distributed systems, communication delays can dominate computation time if not managed properly. Techniques like message aggregation, minimizing communication frequency, and overlapping communication with computation help reduce overheads.
Memory Consistency Models: In shared memory systems, it is important to define rules about how changes made by one processor become visible to others. Models range from strict sequential consistency to relaxed consistency models that allow more optimization but require careful programming.
Synchronization Primitives:
	Mutexes ensure mutual exclusion but may cause contention.
	Semaphores provide signaling mechanisms allowing multiple threads to coordinate.
	Condition variables enable threads to wait for certain conditions before proceeding.
Parallel Algorithm Examples:
	Parallel Sorting: Algorithms like bitonic sort leverage multiple processing elements to sort data faster than sequential sorts.
	Parallel Matrix Multiplication: Dividing matrices into sub-blocks and distributing across processors reduces computation time.
	Graph Processing: Parallel breadth-first search can explore large graphs efficiently by processing multiple frontier nodes simultaneously.
Fault Tolerance Strategies:
	Checkpoint/Restart: Periodically saving execution state allows recovery after failures.
	Replication: Maintaining multiple copies of data or processes ensures availability even if some fail.
	Consensus Algorithms: Protocols like Paxos or Raft enable agreement among distributed nodes despite failures.
Distributed File Systems: Systems like HDFS provide reliable storage across multiple servers with replication and fault tolerance features critical for big data applications.
Security in Distributed Systems: Ensuring data privacy, secure communication channels, authentication, and authorization are essential aspects.
Energy Efficiency: Parallel systems often consume significant energy; optimizing algorithms and hardware for energy savings is an ongoing research area.
Emerging Paradigms:
	Serverless Computing: Abstracts server management allowing developers to focus on functions executed in parallel.
	Edge AI: Deploying AI models at edge devices for real-time processing.
This extended discussion highlights the depth and breadth of parallel distributed computing topics. If you want me to elaborate on specific algorithms, programming examples, architectural details, or case studies, please specify.


"""

# ----------------------
# Split Text
# ----------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([DOCUMENT_TEXT])

# ----------------------
# Embeddings
# ----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------
# Vector Store
# ----------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# ----------------------
# Load LLM Locally (No Token)
# ----------------------
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

# ----------------------
# QA Chain
# ----------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ----------------------
# User Input
# ----------------------
query = st.text_input("Ask a question from the document:")

if query:
    result = qa_chain({"query": query})
    answer = result["result"]
    sources = result["source_documents"]

    if not sources or not answer.strip():
        st.warning("⚠ Question is outside document scope.")
    else:
        st.success(answer)
