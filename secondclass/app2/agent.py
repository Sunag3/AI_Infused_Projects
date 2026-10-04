import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# tiny databse
PRICES = {
    "shoes": 799, "hat": 399, "bag": 1420, "shorts": 1299, "pants": 1699
}

# Step  1 define the tool to extrat the price
# will be a normal python functions
def get_price(item):
    print(f"Tool called: get_price({item})")
    return f"{PRICES.get(item.lower(), 'unknown')}"


# Step 2: describe the tool so model know its exists
tools = [{
    "type": "function",
    "function": {
        "name": "get_price",
        "description": "Get the price of a shop item the user ask about",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": "the item name"}},
            "required": ["item"]
        },
    },
}]


# --- Step 3: the loop - think --> may be call tool -> answer ---
def agent(user_messsage):
    message = [{"role": "user", "content": user_messsage}]

    # send the message + tools menu; the model may ask for a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        tools=tools
    )
    msg = response.choices[0].message

    # if meesage if requesting a tool call
    if msg.tool_calls:
        message.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            print(args)
            result = get_price(args['item'])
            message.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })
        # 4 send everything back so it can answer nicel
        response = client.chat.completions.create(model="gpt-4", messages=message)
        msg = response.choices[0].message

    return msg.content

if __name__=="__main__":
    print(agent("How much are the shoes"))
    print(agent("Hi! how can i help you with"))

