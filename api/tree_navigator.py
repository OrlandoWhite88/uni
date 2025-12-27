import json
import logging
from typing import Dict, List, Any, Optional, Union
from .models import HTSNode

class TreeNavigator:
    """
    Handles all tree navigation and building operations.
    Separated from the main engine for modularity and testability.
    """
    
    def __init__(self):
        self.root = HTSNode({'htsno': '', 'indent': '-1', 'description': 'ROOT', 'superior': None}, node_id=0)
        self.chapters: Dict[str, List[HTSNode]] = {}
        self.code_index: Dict[str, HTSNode] = {}
        self.node_index: Dict[int, HTSNode] = {}
        self.next_node_id = 1
        self.chapters_map = self._init_chapters_map()

    def _init_chapters_map(self) -> Dict[int, str]:
        return {
            1: "Live animals",
            2: "Meat and edible meat offal", 
            3: "Fish and crustaceans, molluscs and other aquatic invertebrates",
            4: "Dairy produce; birds eggs; natural honey; edible products of animal origin, not elsewhere specified or included",
            5: "Products of animal origin, not elsewhere specified or included",
            6: "Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage",
            7: "Edible vegetables and certain roots and tubers",
            8: "Edible fruit and nuts; peel of citrus fruit or melons",
            9: "Coffee, tea, maté and spices",
            10: "Cereals",
            11: "Products of the milling industry; malt; starches; inulin; wheat gluten",
            12: "Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruits; industrial or medicinal plants; straw and fodder",
            13: "Lac; gums, resins and other vegetable saps and extracts",
            14: "Vegetable plaiting materials; vegetable products not elsewhere specified or included",
            15: "Animal or vegetable fats and oils and their cleavage products prepared edible fats; animal or vegetable waxes",
            16: "Preparations of meat, of fish or of crustaceans, molluscs or other aquatic invertebrates",
            17: "Sugars and sugar confectionery",
            18: "Cocoa and cocoa preparations",
            19: "Preparations of cereals, flour, starch or milk; bakers' wares",
            20: "Preparations of vegetables, fruit, nuts or other parts of plants",
            21: "Miscellaneous edible preparations",
            22: "Beverages, spirits and vinegar",
            23: "Residues and waste from the food industries; prepared animal feed",
            24: "Tobacco and manufactured tobacco substitutes",
            25: "Salt; sulfur; earths and stone; plastering materials, lime and cement",
            26: "Ores, slag and ash",
            27: "Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes",
            28: "Inorganic chemicals; organic or inorganic compounds of precious metals, of rare-earth metals, of radioactive elements or of isotopes",
            29: "Organic chemicals",
            30: "Pharmaceutical products",
            31: "Fertilizers",
            32: "Tanning or dyeing extracts; dyes, pigments, paints, varnishes, putty and mastics",
            33: "Essential oils and resinoids; perfumery, cosmetic or toilet preparations",
            34: "Soap, organic surface-active agents, washing preparations, lubricating preparations, artificial waxes, prepared waxes, polishing or scouring preparations, candles and similar articles, modeling pastes, \"dental waxes\" and dental preparations with a basis of plaster",
            35: "Albuminoidal substances; modified starches; glues; enzymes",
            36: "Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations",
            37: "Photographic or cinematographic goods",
            38: "Miscellaneous chemical products",
            39: "Plastics and articles thereof",
            40: "Rubber and articles thereof",
            41: "Raw hides and skins (other than furskins) and leather",
            42: "Articles of leather; saddlery and harness; travel goods, handbags and similar containers; articles of animal gut (other than silkworm gut)",
            43: "Furskins and artificial fur; manufactures thereof",
            44: "Wood and articles of wood; wood charcoal",
            45: "Cork and articles of cork",
            46: "Manufactures of straw, of esparto or of other plaiting materials; basketware and wickerwork",
            47: "Pulp of wood or of other fibrous cellulosic material; waste and scrap of paper or paperboard",
            48: "Paper and paperboard; articles of paper pulp, of paper or of paperboard",
            49: "Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans",
            50: "Silk",
            51: "Wool, fine or coarse animal hair; horsehair yarn and woven fabric",
            52: "Cotton",
            53: "Other vegetable textile fibers; paper yarn and woven fabric of paper yarn",
            54: "Man-made filaments",
            55: "Man-made staple fibers",
            56: "Wadding, felt and nonwovens; special yarns, twine, cordage, ropes and cables and articles thereof",
            57: "Carpets and other textile floor coverings",
            58: "Special woven fabrics; tufted textile fabrics; lace, tapestries; trimmings; embroidery",
            59: "Impregnated, coated, covered or laminated textile fabrics; textile articles of a kind suitable for industrial use",
            60: "Knitted or crocheted fabrics",
            61: "Articles of apparel and clothing accessories, knitted or crocheted",
            62: "Articles of apparel and clothing accessories, not knitted or crocheted",
            63: "Other made up textile articles; sets; worn clothing and worn textile articles; rags",
            64: "Footwear, gaiters and the like; parts of such articles",
            65: "Headgear and parts thereof",
            66: "Umbrellas, sun umbrellas, walking sticks, seatsticks, whips, riding-crops and parts thereof",
            67: "Prepared feathers and down and articles made of feathers or of down; artificial flowers; articles of human hair",
            68: "Articles of stone, plaster, cement, asbestos, mica or similar materials",
            69: "Ceramic products",
            70: "Glass and glassware",
            71: "Natural or cultured pearls, precious or semi-precious stones,precious metals, metals clad with precious metal and articles thereof; imitation jewelry; coin",
            72: "Iron and steel",
            73: "Articles of iron or steel",
            74: "Copper and articles thereof",
            75: "Nickel and articles thereof",
            76: "Aluminum and articles thereof",
            77: "(Reserved for possible future use)",
            78: "Lead and articles thereof",
            79: "Zinc and articles thereof",
            80: "Tin and articles thereof",
            81: "Other base metals; cermets; articles thereof",
            82: "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof of base metal",
            83: "Miscellaneous articles of base metal",
            84: "Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof",
            85: "Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles",
            86: "Railway or tramway locomotives, rolling-stock and parts thereof; railway or tramway track fixtures and fittings and parts thereof; mechanical (including electro-mechanical) traffic signalling equipment of all kinds",
            87: "Vehicles other than railway or tramway rolling stock, and parts and accessories thereof",
            88: "Aircraft, spacecraft, and parts thereof",
            89: "Ships, boats and floating structures",
            90: "Optical, photographic, cinematographic, measuring, checking, precision, medical or surgical instruments and apparatus; parts and accessories thereof",
            91: "Clocks and watches and parts thereof",
            92: "Musical instruments; parts and accessories of such articles",
            93: "Arms and ammunition; parts and accessories thereof",
            94: "Furniture; bedding, mattresses, mattress supports, cushions and similar stuffed furnishings; lamps and lighting fittings, not elsewhere specified or included; illuminated sign illuminated nameplates and the like; prefabricated buildings",
            95: "Toys, games and sports requisites; parts and accessories thereof",
            96: "Miscellaneous manufactured articles",
            97: "Works of art, collectors' pieces and antiques",
            98: "Special classification provisions",
            99: "Temporary legislation; temporary modifications proclaimed pursuant to trade agreements legislation; additional import restrictions proclaimed pursuant to section 22 of the Agricultural Adjustment Act, as amended"
        }

    def build_from_json(self, json_data: Union[str, List[Dict[str, Any]]]) -> None:
        """Build the HTS hierarchy from JSON data. Also build a node index by node_id."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        logging.info(f"Building HTS tree with {len(data)} items")
        parents_by_indent = {-1: self.root}
        
        # Use enumeration to ensure deterministic node IDs based on position
        for index, item in enumerate(data):
            # Node ID is deterministic: position + 1 (since root is 0)
            node_id = index + 1
            node = HTSNode(item, node_id=node_id)
            self.node_index[node_id] = node
            if node.htsno:
                self.code_index[node.htsno] = node
            parent_indent = node.indent - 1
            while parent_indent >= -1:
                if parent_indent in parents_by_indent:
                    parent = parents_by_indent[parent_indent]
                    break
                parent_indent -= 1
            else:
                parent = self.root
            parent.add_child(node)
            parents_by_indent[node.indent] = node
            for indent in list(parents_by_indent.keys()):
                if indent > node.indent:
                    del parents_by_indent[indent]
            
            # Store in chapters mapping - we'll still build this for backward compatibility
            # but won't rely on it for chapter navigation
            if node.htsno:
                chapter = node.get_chapter()
                if chapter:
                    if chapter not in self.chapters:
                        self.chapters[chapter] = []
                    self.chapters[chapter].append(node)
        
        # Update next_node_id to match the number of nodes created
        self.next_node_id = len(data) + 1
        
        # Log verification to ensure consistency
        logging.info(f"Built tree with {len(self.node_index)} nodes, next_node_id={self.next_node_id}")

    def get_node_by_id(self, node_id: int) -> Optional[HTSNode]:
        """Get a node by its ID."""
        return self.node_index.get(node_id)
    
    def get_children(self, node: HTSNode) -> List[HTSNode]:
        """Get the immediate children of a node."""
        if not node:
            return []
        return node.children
        
    def get_chapter_nodes(self, chapter: str) -> List[HTSNode]:
        """Get all nodes for a given chapter."""
        # Don't rely on pre-built chapters index - directly scan the tree
        return [node for node in self.root.children 
                if node.htsno and node.htsno.startswith(chapter)]
    
    def get_chapter_headings(self, chapter: str) -> List[HTSNode]:
        """Get all heading nodes for a given chapter."""
        # Get all top-level nodes that start with this chapter
        chapter_nodes = self.get_chapter_nodes(chapter)
        # Sort them by HTS code to ensure correct order
        return sorted(chapter_nodes, key=lambda n: n.htsno)
    
    def create_chapter_parent(self, chapter: str) -> HTSNode:
        """Create a pseudo-node that has all chapter heading nodes as children."""
        chapter_parent = HTSNode({
            'htsno': chapter,
            'description': self.chapters_map.get(int(chapter), "Unknown chapter"),
            'indent': -1
        }, node_id=-1)
        
        for node in self.get_chapter_headings(chapter):
            chapter_parent.add_child(node)
            
        return chapter_parent
    
    def find_node_by_prefix(self, prefix: str) -> Optional[HTSNode]:
        """
        Find a node that matches the given prefix.
        
        Args:
            prefix: The prefix to search for (chapter or heading number)
            
        Returns:
            The matching node, or None if not found
        """
        # Check if it's an exact match first
        if prefix in self.code_index:
            return self.code_index[prefix]
        
        # If not exact match, try to find nodes with this prefix
        matching_nodes = []
        for code, node in self.code_index.items():
            if code.startswith(prefix):
                matching_nodes.append(node)
        
        if not matching_nodes:
            return None
        
        # Return the shortest matching node (most likely the parent)
        return min(matching_nodes, key=lambda n: len(n.htsno))
