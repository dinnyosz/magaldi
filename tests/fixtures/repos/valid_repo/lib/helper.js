/**
 * Helper functions
 */

function formatName(first, last) {
    return `${first} ${last}`;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

module.exports = { formatName, capitalize };
