"""Unit tests for the Linear sync CLI helpers."""

import scripts.linear_sync as linear_sync


def test_update_issue_state_sends_linear_state_name(monkeypatch):
    """CLI state aliases should be translated before the GraphQL mutation."""
    captured = {}

    def fake_linear_request(query, variables=None):
        captured["query"] = query
        captured["variables"] = variables
        return {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "identifier": "MLO-1",
                    "state": {"name": variables["state"]},
                },
            }
        }

    monkeypatch.setattr(linear_sync, "linear_request", fake_linear_request)

    result = linear_sync.update_issue_state("MLO-1", "done")

    assert result["success"] is True
    assert captured["variables"] == {"identifier": "MLO-1", "state": "Done"}
