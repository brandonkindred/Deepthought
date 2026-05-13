# CLAUDE.md - Deepthought

## Project Overview

Deepthought is a graph-based reasoning engine that uses reinforcement learning (Q-learning) to make predictions from structured and unstructured data. It stores learned weights as edges in a Neo4j graph database, enabling interpretable machine learning without GPUs. The core concept is "elastic vectors" — dynamically constructed weight matrices from graph topology.

## Tech Stack

- **Language:** Java 8
- **Framework:** Spring Boot 2.2.6.RELEASE
- **Database:** Neo4j 4.0+ (graph database, accessed via Spring Data Neo4j / Neo4j OGM)
- **Build:** Apache Maven (use `./mvnw` wrapper if available, otherwise `mvn`)
- **Testing:** TestNG 6.8.8
- **NLP:** Stanford CoreNLP 3.9.1
- **API Docs:** SpringDoc OpenAPI 1.3.0 (Swagger UI at `/swagger-ui.html`)
- **JSON:** org.json 20180130, Gson 2.8.5
- **License:** MIT

## Build Commands

```bash
# Compile
mvn clean compile

# Run tests (only runs tests in the "Regression" TestNG group)
mvn test

# Package as JAR
mvn clean package

# Run the application
mvn spring-boot:run

# Generate OpenAPI spec (runs during integration-test phase)
mvn verify
```

**Main class:** `com.deepthought.App`

## Project Structure

```
src/main/java/
└── com/deepthought/                  # Single root package
    ├── App.java                      # Spring Boot entry point
    ├── api/                          # REST controllers
    │   ├── ReinforcementLearningController.java  # /rl/*
    │   ├── LanguageModelController.java          # /llm/*
    │   ├── LogisticRegressionController.java     # /lr/*
    │   ├── ImageIngestionController.java         # /images/*
    │   ├── TokenSample.java
    │   └── dto/
    │       └── ImageIngestRequest.java
    ├── brain/
    │   ├── Brain.java                # Core prediction/learning orchestrator
    │   ├── QLearn.java               # Q-learning algorithm
    │   ├── TokenVector.java          # Elastic vector construction
    │   ├── Predict.java              # Prediction interface
    │   ├── ActionFactory.java
    │   ├── LanguageModelService.java # Bigram LM over HAS_RELATED_TOKEN
    │   └── LogisticRegressionService.java
    ├── config/
    │   ├── Neo4jConfiguration.java   # Neo4j session/transaction setup
    │   └── ConfigService.java        # (dormant helper, not wired in)
    ├── db/
    │   ├── DataDecomposer.java       # JSON-to-Token decomposition
    │   └── VocabularyWeights.java
    ├── image/
    │   └── ImageProcessingService.java
    ├── models/                       # Neo4j entities
    │   ├── Token.java
    │   ├── Vocabulary.java
    │   ├── MemoryRecord.java
    │   ├── ImageMatrixNode.java
    │   ├── LogisticRegressionModel.java
    │   ├── edges/
    │   │   ├── TokenWeight.java      # HAS_RELATED_TOKEN
    │   │   ├── TokenPolicy.java      # TOKEN_POLICY (dormant)
    │   │   ├── Prediction.java       # PREDICTION
    │   │   └── PartOf.java           # PART_OF
    │   ├── repository/               # Spring Data Neo4j repositories
    │   │   ├── TokenRepository.java  # CRUD + custom Cypher
    │   │   ├── TokenWeightRepository.java
    │   │   ├── VocabularyRepository.java
    │   │   ├── MemoryRecordRepository.java
    │   │   ├── PredictionRepository.java
    │   │   ├── ImageMatrixNodeRepository.java
    │   │   ├── PartOfRepository.java
    │   │   └── LogisticRegressionModelRepository.java
    │   └── services/
    │       └── TokenService.java
    └── observableStructs/            # Concurrent data structures
        ├── ConcurrentNode.java
        ├── ObservableHash.java
        └── ObservableQueue.java

src/test/java/
└── com/deepthought/                  # Tests mirror prod packages
    ├── BrainTests.java
    ├── DataDecomposerTests.java
    ├── LanguageModelServiceTests.java
    ├── LogisticRegressionServiceTests.java
    ├── TokenVectorTests.java
    ├── api/
    │   ├── ReinforcementLearningControllerTests.java
    │   ├── ImageIngestionControllerTests.java
    │   └── dto/ImageIngestRequestTests.java
    ├── image/
    │   └── ImageProcessingServiceTests.java
    ├── models/                       # Entity tests
    │   ├── VocabularyTests.java
    │   ├── MemoryRecordTokenTests.java
    │   ├── ImageMatrixNodeTests.java
    │   └── edges/PartOfTests.java
    ├── observableStructs/
    │   ├── ConcurrentNodeTests.java
    │   ├── ObservableHashTests.java
    │   ├── ObservableQueueTests.java
    │   └── ObservableStructsTests.java
    └── resourceClasses/
        └── SelfContainedTestObject.java
```

## Architecture

### Package Namespace

All production code lives under the single root package `com.deepthought.*`.
`App.java` configures `@ComponentScan(basePackages = {"com.deepthought"})`
(redundant with `@SpringBootApplication`'s implicit scan from the same package,
but kept explicit).

### Neo4j Graph Schema

```
(Token) -[HAS_RELATED_TOKEN {weight}]-> (Token)
(MemoryRecord) -[PREDICTION {weight}]-> (Token)
```

- **Token** nodes hold a `value` string (e.g., "button", "click")
- **TokenWeight** edges store learned weights (0.0–1.0) between tokens
- **MemoryRecord** nodes store prediction context (inputs, outputs, policy matrix)
- **Prediction** edges link memory records to predicted tokens with confidence weights

### Request Flow

1. Client sends JSON to `POST /rl/predict` with input data and output token labels
2. `DataDecomposer` breaks JSON into a list of `Token` objects
3. `Brain.generatePolicy()` builds a weight matrix from graph edges (or initializes random weights)
4. `Brain.predict()` sums columns and normalizes to produce a probability distribution
5. A `MemoryRecord` is persisted with the policy matrix and prediction
6. Client provides feedback via `POST /rl/learn` with the memory ID and actual token
7. `Brain.learn()` applies Q-learning updates to edge weights in the graph

### Q-Learning Parameters

- Learning rate: 0.1
- Discount factor: 0.1
- Reward structure: +2 (exact match), +1 (partial match), -1 (wrong prediction), -2 (unrelated)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/rl/predict` | Generate prediction from JSON input + output labels |
| POST | `/rl/learn` | Apply learning feedback for a memory record |
| POST | `/rl/train` | Train on labeled JSON data |

## Configuration

**`src/main/resources/application.properties`** — Neo4j connection settings must be configured:

```properties
spring.data.neo4j.uri=bolt://localhost:7687
spring.data.neo4j.username=neo4j
spring.data.neo4j.password=<password>
```

The properties file is templated with commented-out placeholders. Neo4j must be running before starting the application.

## Code Conventions

- **Naming:** snake_case for variables and parameters (`token_repo`, `input_tokens`, `memory_id`), PascalCase for classes, camelCase for methods
- **Dependency injection:** Field-level `@Autowired` injection throughout
- **Logging:** SLF4J with Logback — `private static Logger log = LoggerFactory.getLogger(ClassName.class)`
- **Neo4j queries:** Custom Cypher via `@Query` annotations on repository interfaces
- **API documentation:** OpenAPI 3 annotations (`@Operation`, `@Schema`) on controllers and models
- **Entity mapping:** Neo4j OGM annotations (`@NodeEntity`, `@RelationshipEntity`, `@Id`, `@GeneratedValue`)
- **JSON serialization:** Jackson with `@JsonIgnore` to exclude graph relationships from API responses
- **Validation:** `@NotBlank` on required entity fields
- **Error handling:** Exceptions propagated from controller methods; custom exception classes in controller files
- **Brace style:** Opening braces on same line for classes/methods, K&R style
- **Indentation:** Tabs (not spaces)

## Testing

- Tests use **TestNG** (not JUnit) with `@Test` annotations
- Maven Surefire is configured to only run tests in the `Regression` group
- Tests must be annotated with `@Test(groups = "Regression")` to be picked up by `mvn test`
- Current test coverage is minimal — most test methods are stubs
- Test packages mirror source under `com.deepthought.*`

## Dependencies (External Services)

- **Neo4j 4.0+** — Required. Must be running and accessible via Bolt protocol
- No CI/CD pipelines are configured
- No Docker configuration for the application itself (README references Docker for Neo4j setup)

## Key Files for Common Tasks

- **Adding a new API endpoint:** `src/main/java/com/deepthought/api/`
- **Modifying prediction/learning logic:** `src/main/java/com/deepthought/brain/Brain.java`
- **Adding a new entity/relationship:** `src/main/java/com/deepthought/models/` and `src/main/java/com/deepthought/models/edges/`
- **Adding a new repository query:** `src/main/java/com/deepthought/models/repository/`
- **Changing Neo4j config:** `src/main/java/com/deepthought/config/Neo4jConfiguration.java`
- **Modifying JSON decomposition:** `src/main/java/com/deepthought/db/DataDecomposer.java`
- **Application properties:** `src/main/resources/application.properties`
