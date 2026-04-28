import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def create_entity(tx, name, entity_type):
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.type = $type
        """,
        name=name,
        type=entity_type
    )


def create_relationship(tx, source, relationship, target, evidence):
    rel_type = relationship.upper().replace(" ", "_").replace("-", "_")

    query = f"""
    MATCH (a:Entity {{name: $source}})
    MATCH (b:Entity {{name: $target}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r.evidence = $evidence
    """

    tx.run(query, source=source, target=target, evidence=evidence)


def load_graph(graph_file="data/graphs/kg.json"):
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with driver.session() as session:
        for chunk in data:

            for entity in chunk.get("entities", []):
                session.execute_write(
                    create_entity,
                    entity["name"],
                    entity.get("type", "Unknown")
                )

            for rel in chunk.get("relationships", []):
                try:
                    session.execute_write(
                        create_relationship,
                        rel["source"],
                        rel["relationship"],
                        rel["target"],
                        rel.get("evidence", "")
                    )
                except Exception as e:
                    print(f"Skipping relationship: {e}")

    print("✅ Graph loaded into Neo4j!")


if __name__ == "__main__":
    load_graph()
    driver.close()
