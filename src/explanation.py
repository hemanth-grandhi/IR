class Explainer:
    @staticmethod
    def format_explanation(doc_id, metadata, score, contributions, max_terms=3):
        """Formats an explainable IR output for a single document."""
        title = metadata.get('title', 'Unknown Title')
        
        output = [
            f"Document: {title} (ID: {doc_id})",
            f"Score: {score:.4f}",
            "Why this ranked highly:"
        ]
        
        # Sort terms by their contribution to the final score
        sorted_terms = sorted(contributions.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for term, scores in sorted_terms[:max_terms]:
            total = scores['total']
            t_score = scores['title_score']
            b_score = scores['body_score']
            
            if total > 0.3:
                impact = "high"
            elif total > 0.1:
                impact = "medium"
            else:
                impact = "low"
                
            zone_info = []
            if t_score > 0: zone_info.append("title")
            if b_score > 0: zone_info.append("body")
            zone_str = f" (matched in: {', '.join(zone_info)})" if zone_info else ""
            
            output.append(f"  - '{term}' -> {impact} contribution ({total:.4f}){zone_str}")
            
        return "\n".join(output)
