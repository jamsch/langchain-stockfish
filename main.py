from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, tool
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from stockfish import Stockfish
import chess.pgn
import io
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


stockfish = Stockfish(path="./stockfish_binary/stockfish", depth=10)
stockfish.set_elo_rating(1600)
board = chess.Board()


@tool
def fish(query: str) -> str:
    """Evaluates a current position using stockfish."""
    try:
        # if query starts with "1. e4 e5 2. Nc3 Nc6 3. d4"
        # then we need to set the position to the start of the game
        if query.startswith("1."):
            pgn = io.StringIO(query)
            game = chess.pgn.read_game(pgn)
            board.reset()
            board.set_fen(game.board().fen())
        # else if it contains two numbers, e.g. d7d5, then it's in uci format
        elif len(query) == 4 and query[0].isalpha() and query[1].isnumeric() and query[2].isalpha() and query[3].isnumeric():
            move = chess.Move.from_uci(query)
            board.push(move)
        else:
            # Split the query by spaces and push each move
            for move in query.split(" "):
                board.push_san(move)
    except chess.IllegalMoveError as e:
        return f"Illegal move. Tell the user that the move they just made was illegal."
    except chess.InvalidMoveError as e:
        return f"Invalid move. Make sure you're relaying a single move using standard algebraic notation, for example Bc4 or e4 or Qh5."

    fen = board.fen()
    stockfish.set_fen_position(fen)

    top_moves = stockfish.get_top_moves(1)

    # append { is_capture: stockfish.will_move_be_a_capture(move.Move) } to top_moves
    top_moves_with_capture = [
        {**move,
            # Move is a string like "e2e4"
            "from": move["Move"][:2],
            "to": move["Move"][2:],
            "san_move": board.san(chess.Move.from_uci(move["Move"])),
            "evaluation": move["Centipawn"] and move["Centipawn"] / 100 or f'mate in {move["Mate"]}',
            "piece_name_from": stockfish.get_what_is_on_square(move["Move"][:2]).name,
            "piece_name_to": stockfish.get_what_is_on_square(move["Move"][2:]),
            "is_capture": stockfish.will_move_be_a_capture(move["Move"]).name
         }
        for move in top_moves
    ]

    top_moves_formatted_as_string = ", ".join(
        [f"{move['san_move']} {' capturing ' + move['piece_name_to'].name if move.get('piece_name_to') else ''} (eval: {move['evaluation']})" for move in top_moves_with_capture])

    evaluation = stockfish.get_evaluation()
    evaluation_value = evaluation["value"] / 100
    evaluation_text = ""
    if evaluation["type"] == "cp":
        evaluation_text = f"{'+' if evaluation_value >= 0 else '-'}{abs(evaluation_value)}"
    else:
        evaluation_text = "a mate in "

    whos_winning = evaluation_value >= 0 and "White" or "Black"

    # do the move
    move = chess.Move.from_uci(top_moves[0]["Move"])
    board.push(move)

    # Board visual:\n{stockfish.get_board_visual()}
    return f"""{whos_winning} is winning with {evaluation_text}. I will proceed with the top move: {top_moves_formatted_as_string.strip()}"""


prefix = """You are a very disrespectful chess player that secretly uses Stockfish to find the best move. You will reply in rude remarks to the player when they make blunders. You have access to the following tools:"""
suffix = """Begin! Remember: You can only relay the user's move to Stockfish in standard algebraic notation ONCE. When answering, tell your opponent the move that you just made with the help of Stockfish while taunting the user in a clever, witty way. NEVER mention Stockfish."""


tools = [
    Tool(
        name="Stockfish",
        func=fish.run,
        description="Relay the user's move here in the format: e4, Qh5, Qxf7, Nc6, etc."
    )
]

system_prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=[]
)

messages = [
    SystemMessagePromptTemplate(prompt=system_prompt),
    HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
                                             f"(but I haven't seen any of it! I only see what "
                                             "you return as final answer):\n{agent_scratchpad}")
]

full_prompt = ChatPromptTemplate.from_messages(messages)

memory = ConversationBufferMemory(
    input_key="input",
)

llm_chain = LLMChain(llm=OpenAI(temperature=0),
                     prompt=full_prompt, memory=memory)

tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# cli prompt user to start the conversation

print("GrouchyGM: Let's play chess! You'll be playing as white. I'll be playing as black.")

while True:
    # "I'll be playing as white. Let's start with 1. e4 e5, 2. Nc3 Nc6 3. d4"
    message = input("You: ")
    if not message:
        print("GrouchyGM: make a move!")
        continue
    result = agent_executor.run(message)
    print(f"GrouchyGM: {result}")
