#!/usr/bin/env python3
"""
Hierarchy Analyzer - Build and analyze component hierarchy tree.

This module provides functionality to build a hierarchy tree from diagram JSON
and analyze it for bottom-up placement processing.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class HierarchyNode:
    """Represents a node in the hierarchy tree."""
    id: str
    type: str  # "component", "container"
    children: List['HierarchyNode'] = field(default_factory=list)
    parent: Optional['HierarchyNode'] = None
    level: int = 0  # Depth in hierarchy (0 = root)
    data: Dict[str, Any] = field(default_factory=dict)  # Original data from JSON


class HierarchyAnalyzer:
    """Analyze and build component hierarchy."""
    
    def __init__(self, diagram_json: Dict[str, Any]):
        """
        Initialize hierarchy analyzer.
        
        Args:
            diagram_json: Diagram JSON definition
        """
        self.diagram_json = diagram_json
        self.root = None
        self.nodes_by_id: Dict[str, HierarchyNode] = {}
        self._build_tree()
    
    def _build_tree(self):
        """Build hierarchy tree from diagram JSON."""
        # Create nodes for all containers
        container_defs = {c["id"]: c for c in self.diagram_json.get("containers", [])}
        
        # Create container nodes
        for container_id, container_def in container_defs.items():
            node = HierarchyNode(
                id=container_id,
                type="container",
                data=container_def
            )
            self.nodes_by_id[container_id] = node
        
        # Create component nodes (for components not in containers)
        all_contained_ids = set()
        for container_def in container_defs.values():
            all_contained_ids.update(container_def.get("contains", []))
        
        # Create root node for top-level elements
        self.root = HierarchyNode(id="root", type="root")
        
        # Add top-level components to root
        for comp_def in self.diagram_json.get("components", []):
            comp_id = comp_def["id"]
            if comp_id not in all_contained_ids:
                node = HierarchyNode(
                    id=comp_id,
                    type="component",
                    data=comp_def
                )
                self.nodes_by_id[comp_id] = node
                self.root.children.append(node)
                node.parent = self.root
                node.level = 0
        
        # Add top-level containers to root
        for container_id, container_def in container_defs.items():
            # Check if this container is nested in another container
            is_nested = False
            for other_id, other_def in container_defs.items():
                if container_id in other_def.get("contains", []):
                    is_nested = True
                    break
            
            if not is_nested:
                container_node = self.nodes_by_id[container_id]
                self.root.children.append(container_node)
                container_node.parent = self.root
                container_node.level = 0
        
        # Build parent-child relationships
        for container_id, container_def in container_defs.items():
            container_node = self.nodes_by_id[container_id]
            
            # Add child components
            for child_id in container_def.get("contains", []):
                if child_id in self.nodes_by_id:
                    child_node = self.nodes_by_id[child_id]
                    container_node.children.append(child_node)
                    child_node.parent = container_node
                    child_node.level = container_node.level + 1
                else:
                    # Child is a component, create node for it
                    child_node = HierarchyNode(
                        id=child_id,
                        type="component",
                        data={}
                    )
                    self.nodes_by_id[child_id] = child_node
                    container_node.children.append(child_node)
                    child_node.parent = container_node
                    child_node.level = container_node.level + 1
        
        # Recursively set levels
        self._set_levels(self.root, 0)
    
    def _set_levels(self, node: HierarchyNode, level: int):
        """Recursively set levels for all nodes."""
        node.level = level
        for child in node.children:
            self._set_levels(child, level + 1)
    
    def get_container_levels(self) -> List[List[str]]:
        """
        Get containers organized by depth level.
        
        Returns:
            List of lists, where each inner list contains container IDs at that depth
        """
        levels: Dict[int, List[str]] = {}
        
        def collect_levels(node: HierarchyNode):
            if node.type == "container":
                if node.level not in levels:
                    levels[node.level] = []
                levels[node.level].append(node.id)
            
            for child in node.children:
                collect_levels(child)
        
        collect_levels(self.root)
        
        # Return sorted by level (deepest first for bottom-up processing)
        max_level = max(levels.keys()) if levels else -1
        result = []
        for level in range(max_level, -1, -1):
            if level in levels:
                result.append(levels[level])
        
        return result
    
    def get_top_level_elements(self) -> List[str]:
        """
        Get all top-level element IDs (containers and components).
        
        Returns:
            List of element IDs at root level
        """
        return [child.id for child in self.root.children]
    
    def get_container_children(self, container_id: str) -> List[str]:
        """
        Get child component IDs for a container.
        
        Args:
            container_id: Container ID
            
        Returns:
            List of child component IDs
        """
        if container_id not in self.nodes_by_id:
            return []
        
        container_node = self.nodes_by_id[container_id]
        return [child.id for child in container_node.children if child.type == "component"]
    
    def get_container_definition(self, container_id: str) -> Optional[Dict[str, Any]]:
        """Get container definition from JSON."""
        for container_def in self.diagram_json.get("containers", []):
            if container_def["id"] == container_id:
                return container_def
        return None
    
    def find_connected_groups(
        self, 
        component_ids: List[str], 
        connections: List[Any]
    ) -> List[List[str]]:
        """
        Find connected component groups using graph connectivity.
        
        Args:
            component_ids: List of component IDs to analyze
            connections: List of Connection objects
            
        Returns:
            List of component groups, where each group is a list of connected component IDs
        """
        # Build adjacency list
        graph: Dict[str, Set[str]] = {comp_id: set() for comp_id in component_ids}
        
        for conn in connections:
            from_id = conn.from_component_id
            to_id = conn.to_component_id
            
            if from_id in graph and to_id in graph:
                graph[from_id].add(to_id)
                graph[to_id].add(from_id)
        
        # Find connected components using DFS
        visited: Set[str] = set()
        groups: List[List[str]] = []
        
        def dfs(comp_id: str, group: List[str]):
            if comp_id in visited:
                return
            visited.add(comp_id)
            group.append(comp_id)
            
            for neighbor in graph[comp_id]:
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for comp_id in component_ids:
            if comp_id not in visited:
                group = []
                dfs(comp_id, group)
                if group:
                    groups.append(group)
        
        return groups

