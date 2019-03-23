# Q Learning AI

This contains an AI that is be trained using Q Learning.
It can simulate simple grid games.
Several games have been included in **/game_templates**.
You can create your own template file to train AI on new games.

## Usage
Run simulation

```bash
python3 ql_ai.py
```
Change the template file loaded to simulate different games.
```python
# Load template
    with open('./game_templates/race_track_template.json') as json_file:
        json_str = json_file.read()
        game_data = json.loads(json_str)
```
- Lizard Game:
- 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
