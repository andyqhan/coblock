function parseGameLogs(logContent) {
    const trials = [];
    const lines = logContent.split('\n');
    const timestampRegex = /^\d{4}-\d{2}-\d{2}/;

    let currentTrial = null;
    let currentTurn = null;
    let collectingResponse = false;
    let responseLines = [];

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const hasTimestamp = timestampRegex.test(line);

        // If we hit a timestamp while collecting response, finalize the response
        if (hasTimestamp && collectingResponse && currentTurn) {
            currentTurn.response = responseLines.join('\n').trim();
            parseAction(currentTurn);
            collectingResponse = false;
            responseLines = [];
        }

        // Parse trial start
        const trialMatch = line.match(/\[(\d+)\/\d+\] Running comparison\.\.\./);
        if (trialMatch) {
            if (currentTrial) {
                // Finalize any pending turn
                if (currentTurn) {
                    currentTrial.turns.push(currentTurn);
                    currentTurn = null;
                }
                trials.push(currentTrial);
            }
            currentTrial = {
                trial: parseInt(trialMatch[1]),
                agent1: null,
                agent2: null,
                structure: null,
                total_turns: 0,
                success: false,
                voted_to_end: false,
                turns: []
            };
            continue;
        }

        // Parse trial metadata
        if (currentTrial && !currentTrial.agent1) {
            const agentMatch = line.match(/Running trial \d+ for (.+) vs (.+)/);
            if (agentMatch) {
                currentTrial.agent1 = agentMatch[1];
                currentTrial.agent2 = agentMatch[2];
            }
            i++;
            line = lines[i];
            const structureMatch = line.match(/Structure: (.+)/);
            if (structureMatch) {
                currentTrial.structure = structureMatch[1];
            }
            continue;
        }

        // Parse turn start
        const turnMatch = line.match(/=== Turn (\d+): (\w+)'s turn ===/);
        if (turnMatch && currentTrial) {
            // Save previous turn if exists
            if (currentTurn) {
                currentTrial.turns.push(currentTurn);
            }

            currentTurn = {
                agent: turnMatch[2],
                action: null,
                response: '',
                success: false,
                block_pos: null,
                block_color: null,
                chat_recipient: null,
                chat_content: null
            };
            currentTrial.total_turns = parseInt(turnMatch[1]);
        }

        // Collect agent response
        if (line.includes(' response: ')) {
            collectingResponse = true;
            const responseStart = line.indexOf(' response: ') + ' response: '.length;
            responseLines.push(line.substring(responseStart));
            continue;
        }

        if (collectingResponse && !hasTimestamp) {
            responseLines.push(line);
            continue;
        }

        // Parse action results
        if (currentTurn && hasTimestamp) {
            // Place block success
            const placeSuccessMatch = line.match(/(\w+) successfully placed (\w+) block at \(([^)]+)\)/);
            if (placeSuccessMatch && placeSuccessMatch[1] === currentTurn.agent) {
                currentTurn.success = true;
                currentTurn.block_color = placeSuccessMatch[2];
                currentTurn.block_pos = placeSuccessMatch[3];
            }

            // Remove block success
            const removeSuccessMatch = line.match(/(\w+) successfully removed block at \(([^)]+)\)/);
            if (removeSuccessMatch && removeSuccessMatch[1] === currentTurn.agent) {
                currentTurn.success = true;
                currentTurn.block_pos = removeSuccessMatch[2];
            }

            // Chat message
            const chatMatch = line.match(/(\w+) sent message to (\w+): (.+)/);
            if (chatMatch && chatMatch[1] === currentTurn.agent) {
                currentTurn.chat_recipient = chatMatch[2];
                currentTurn.chat_content = chatMatch[3];
            }

            // Wait
            if (line.includes(currentTurn.agent + ' waited')) {
                currentTurn.action = 'wait()';
            }

            // Multiple actions warning
            if (line.includes(currentTurn.agent + ' tried to execute multiple world actions')) {
                // The action was attempted but only first was executed
            }

            // Action failure
            if (line.includes('Failed to place block') || line.includes('Failed to remove block')) {
                currentTurn.success = false;
            }
        }

        // Parse trial completion
        if (currentTrial && hasTimestamp) {
            if (line.includes('ALL AGENTS VOTED TO END GAME')) {
                currentTrial.voted_to_end = true;
            }

            const completionMatch = line.match(/Trial completed: (SUCCESS|FAILURE) in (\d+) turns/);
            if (completionMatch) {
                currentTrial.success = completionMatch[1] === 'SUCCESS';
                currentTrial.total_turns = parseInt(completionMatch[2]);
            }

            if (line.includes('Game ended after 30 turns without achieving goal')) {
                currentTrial.success = false;
                currentTrial.total_turns = 30;
            }
        }
    }

    // Finalize any pending turn
    if (currentTurn && currentTrial) {
        if (collectingResponse) {
            currentTurn.response = responseLines.join('\n').trim();
            parseAction(currentTurn);
        }
        currentTrial.turns.push(currentTurn);
    }

    // Add the last trial
    if (currentTrial) {
        trials.push(currentTrial);
    }

    return trials;
}

function parseAction(turn) {
    const response = turn.response;

    // Parse place_block
    const placeMatch = response.match(/place_block\s*\(\s*block_type\s*=\s*"?(\w+)"?\s*,\s*pos\s*=\s*\("?([^)]+)"?\)\s*\)/);
    if (placeMatch) {
        turn.action = `place_block(block_type=${placeMatch[1]}, pos=(${placeMatch[2]}))`;
        turn.block_color = placeMatch[1];
        turn.block_pos = placeMatch[2].replace(/["\s]/g, '');
        return;
    }

    // Parse remove_block
    const removeMatch = response.match(/remove_block\s*\(\s*pos\s*=\s*\(([^)]+)\)\s*\)/);
    if (removeMatch) {
        turn.action = `remove_block(pos=(${removeMatch[1]}))`;
        turn.block_pos = removeMatch[1].replace(/\s/g, '');
        return;
    }

    // Parse wait
    const waitMatch = response.match(/wait\s*\(\s*\)/);
    if (waitMatch) {
        turn.action = 'wait()';
        return;
    }

    // Parse end_game
    const endMatch = response.match(/end_game\s*\(\s*\)/);
    if (endMatch) {
        turn.action = 'end_game()';
        return;
    }

    // Parse send_chat
    const chatMatch = response.match(/send_chat\s*\(\s*to\s*=\s*"([^"]+)"\s*,\s*message\s*=\s*"([^"]+)"\s*\)/);
    if (chatMatch) {
        turn.action = `send_chat(to="${chatMatch[1]}", message="${chatMatch[2]}")`;
        turn.chat_recipient = chatMatch[1];
        turn.chat_content = chatMatch[2];
        return;
    }

    // If no action found
    turn.action = 'unknown';
}

function getGoal(structure) {
    if (structure.includes("/")) {
        structure = structure.split("/")[1];
    }
    if (structure.includes(".")) {
        structure = structure.split(".")[0];
    }
    const goals = {
        bridge: {
            "[0,0,0]": "gray",
            "[0,0,1]": "gray",
            "[0,0,2]": "gray",
            "[1,0,2]": "brown",
            "[2,0,2]": "brown",
            "[3,0,2]": "brown",
            "[4,0,2]": "brown",
            "[5,0,0]": "gray",
            "[5,0,1]": "gray",
            "[5,0,2]": "gray"
        }
    };

    return goals[structure.toLowerCase()] || {};
}
