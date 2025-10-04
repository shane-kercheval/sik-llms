## Summary
- add Azure OpenAI as a first-class provider while preserving existing registry ergonomics
- document configuration and testing expectations for Azure usage

## Checklist
- [x] Extend provider registry and model metadata for Azure OpenAI deployments
- [x] Implement Azure client wrappers supporting key features (chat, tools, reasoning, structured output, Responses API)
- [x] Provide configuration defaults/env-var wiring for API keys, endpoints, deployments, and API versions
- [x] Add unit and optional integration tests covering Azure behaviour
- [x] Update README/docs with Azure setup, examples, and testing guidance
- [x] QA: linting/tests updated or added for Azure paths
