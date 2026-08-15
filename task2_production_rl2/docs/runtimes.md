# Runtime and sandbox boundaries

- `subprocess` is used for trusted Verifiers orchestration inside a disposable Modal
  container. It is not called a sandbox.
- The policy has no raw subprocess tool. Its MCP calls create and control an E2B microVM,
  which is the untrusted code boundary.
- E2B uses `secure=True`, internet disabled by default, a 900-second lifetime, 120-second
  command timeout, bounded returned output, and a kill callback.
- The API key exists only in the Modal parent and a mode-0600 key file read by the MCP
  child. It is not copied into the E2B guest.
- Safe paths may be relative to `/home/user/workspace` or absolute beneath it. Traversal
  and other absolute roots are rejected by tests.

Docker is absent on the Windows host and is not used as a substitute. The requested
production isolation is E2B inside Modal; the real Modal→E2B smoke is the runtime evidence.
