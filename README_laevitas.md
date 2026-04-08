Laevitas setup

1. Create a local .env file in the project root.
2. Add your API key to the file using this format:
   LAEVITAS_API_KEY=your_key_here
3. Optional settings:
   LAEVITAS_BASE_URL=https://apiv2.laevitas.ch
   LAEVITAS_DEFAULT_ENDPOINT=/your/default/endpoint
4. Test with:
   python laevitas_test.py /your/endpoint
   python laevitas_test.py /your/endpoint --param key=value

Notes
- The .env file is ignored by git.
- The client sends the key in the apikey header.
